#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master-Runner für Winterthur Stadtbaum v6
=========================================

Startet viele Szenarien aus einer YAML-Datei, legt je Szenario einen eigenen
Output-Ordner an und sammelt zentrale Kennzahlen in scenario_summary.csv.
Optional wird zusätzlich eine SQLite-Datenbank geschrieben.

Beispiel:
    python scripts/run_winterthur_scenarios_v6.py --scenario_config configs/scenario_grid_v6.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def require_yaml():
    if yaml is None:
        raise RuntimeError("PyYAML fehlt. Installation: conda install pyyaml oder pip install pyyaml")


def load_yaml(path: str | Path) -> dict:
    require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_slug(value: object, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return s[:max_len] or "run"


def expand_grid(grid: Dict[str, Iterable[object]]) -> List[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def build_runs(cfg: dict) -> List[dict]:
    base = cfg.get("base", {}) or {}
    grid_runs = expand_grid(cfg.get("grid", {}) or {})
    explicit_runs = cfg.get("runs", []) or []

    runs = []
    for i, override in enumerate(grid_runs, start=1):
        merged = {**base, **override}
        merged.setdefault("run_id", f"grid_{i:03d}")
        runs.append(merged)

    start = len(runs) + 1
    for j, override in enumerate(explicit_runs, start=start):
        merged = {**base, **override}
        merged.setdefault("run_id", f"run_{j:03d}")
        runs.append(merged)

    return runs


def write_run_config(run_dir: Path, params: dict) -> Path:
    require_yaml()
    path = run_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, allow_unicode=True, sort_keys=True)
    return path


def read_summary_for_run(run_dir: Path, params: dict) -> dict:
    row = {
        "run_id": params.get("run_id"),
        "out_dir": str(run_dir),
        "status": "ok",
    }

    # Parameter aufnehmen
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"param_{k}"] = v
        else:
            row[f"param_{k}"] = json.dumps(v, ensure_ascii=False)

    total_m = run_dir / "total_milestones.csv"
    total_s = run_dir / "total_summary.csv"
    src = total_m if total_m.exists() else total_s
    if src.exists():
        df = pd.read_csv(src)
        for _, r in df.iterrows():
            off = int(r["year_offset"])
            for col in ["mean", "p05", "p25", "p50", "p75", "p95", "min", "max"]:
                if col in r:
                    row[f"y{off}_{col}"] = r[col]
            if "year" in r:
                row[f"y{off}_calendar_year"] = int(r["year"])
    else:
        row["status"] = "missing_summary"

    target = run_dir / "target_summary.csv"
    if target.exists():
        try:
            t = pd.read_csv(target)
            for _, r in t.iterrows():
                row[f"target_{r['Kennzahl']}"] = r["Wert"]
        except Exception as e:
            row["target_summary_error"] = str(e)

    return row


def write_sqlite(db_path: Path, summary_df: pd.DataFrame, run_rows: List[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        summary_df.to_sql("scenario_summary", con, if_exists="replace", index=False)

        params_rows = []
        for row in run_rows:
            run_id = row.get("run_id")
            for k, v in row.items():
                if k.startswith("param_"):
                    params_rows.append({
                        "run_id": run_id,
                        "parameter": k.replace("param_", "", 1),
                        "value": v,
                    })
        if params_rows:
            pd.DataFrame(params_rows).to_sql("parameters_long", con, if_exists="replace", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Startet viele Winterthur-v6-Szenarien und aggregiert Ergebnisse.")
    ap.add_argument("--scenario_config", required=True, help="YAML-Datei mit base/grid/runs.")
    ap.add_argument("--script", default=None, help="Pfad zum v6-Modellskript. Überschreibt Wert aus YAML.")
    ap.add_argument("--output_root", default=None, help="Basisordner für alle Runs. Überschreibt Wert aus YAML.")
    ap.add_argument("--sqlite", default=None, help="Optionaler SQLite-Dateipfad. Überschreibt Wert aus YAML.")
    ap.add_argument("--dry_run", action="store_true", help="Nur geplante Runs ausgeben, nichts starten.")
    args = ap.parse_args()

    cfg_path = Path(args.scenario_config)
    cfg = load_yaml(cfg_path)

    script = Path(args.script or cfg.get("script", "scripts/winterthur_tree_stochastic_goal_planning_v6.py"))
    output_root = Path(args.output_root or cfg.get("output_root", "output/scenario_runs"))
    sqlite_path = args.sqlite if args.sqlite is not None else cfg.get("sqlite_path")

    if not script.exists():
        raise FileNotFoundError(f"Modellskript nicht gefunden: {script}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = safe_slug(cfg.get("batch_name", "batch"))
    batch_dir = output_root / f"{stamp}_{batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(cfg)
    print(f"Geplante Runs: {len(runs)}")
    print(f"Batch-Ordner: {batch_dir}")

    summary_rows = []

    for i, params in enumerate(runs, start=1):
        # lesbare Run-ID erzeugen
        interesting = []
        for k in ["annual_new_trees", "new_tree_strategy", "replacement_rate", "replacement_delay", "tree_q", "tree_k", "climate_trend_end"]:
            if k in params:
                interesting.append(f"{k}-{params[k]}")
        rid_base = params.get("run_id") or f"run_{i:03d}"
        run_id = safe_slug(str(rid_base) + ("_" + "_".join(interesting) if interesting else ""))
        params["run_id"] = run_id

        run_dir = batch_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        params["out_dir"] = str(run_dir)
        params["auto_timestamp_outdir"] = False

        cfg_file = write_run_config(run_dir, params)
        cmd = [sys.executable, str(script), "--config", str(cfg_file)]

        print(f"\n[{i}/{len(runs)}] {run_id}")
        print(" ".join(cmd))

        if args.dry_run:
            row = {"run_id": run_id, "out_dir": str(run_dir), "status": "dry_run"}
            summary_rows.append(row)
            continue

        log_path = run_dir / "runner_stdout_stderr.log"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            row = {
                "run_id": run_id,
                "out_dir": str(run_dir),
                "status": f"failed_returncode_{proc.returncode}",
                "log": str(log_path),
            }
            row.update({f"param_{k}": v for k, v in params.items() if isinstance(v, (str, int, float, bool)) or v is None})
        else:
            row = read_summary_for_run(run_dir, params)

        summary_rows.append(row)

        # Zwischenstand nach jedem Run speichern
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(batch_dir / "scenario_summary.csv", index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = batch_dir / "scenario_summary.csv"
    summary_xlsx = batch_dir / "scenario_summary.xlsx"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="scenario_summary")

    if sqlite_path:
        db_path = Path(sqlite_path)
        if not db_path.is_absolute():
            db_path = batch_dir / db_path
        write_sqlite(db_path, summary_df, summary_rows)
        print(f"SQLite geschrieben: {db_path}")

    print("\nFertig.")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary XLSX: {summary_xlsx}")


if __name__ == "__main__":
    main()
