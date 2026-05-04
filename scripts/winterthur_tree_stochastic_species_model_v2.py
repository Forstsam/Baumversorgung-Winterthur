#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastisches Stadtbaum-Bestandsmodell v2
-----------------------------------------
Erweiterte Version des artenweisen Monte-Carlo-Modells.

Neu gegenüber v1:
- optionale artspezifische Multiplikatoren aus externer CSV/XLSX
- optionale kategoriespezifische Multiplikatoren
- optionaler Ersatzarten-Plan (replacement species map)
- optionaler Klimatrend als Start/Ende statt nur fixer jährlicher Steigung
- zusätzliche Ausgaben nach Kategorie und Altersklasse
- Jahreskennzahlen zu Ausfällen, Ersatzpflanzungen und Nettoveränderung
- freier Satz von Meilensteinjahren

Ziel:
- zukünftigen Baumbestand unter Unsicherheit simulieren
- Resultate nach Art, Kategorie und Gesamtbestand ausgeben
- Parameter flexibel steuerbar machen

Hinweis:
Das Modell liefert Szenarien, keine exakten Einzelfallprognosen.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# I/O und Hilfsfunktionen
# -----------------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("Unterstützte Formate: CSV, XLSX, XLS")


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(arr, lo), hi)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summarize_percentiles(values: np.ndarray, probs=(0.05, 0.5, 0.95)) -> Dict[str, float]:
    if len(values) == 0:
        return {f"q{int(p*100):02d}": np.nan for p in probs}
    q = np.quantile(values, probs)
    return {f"q{int(p*100):02d}": float(v) for p, v in zip(probs, q)}


def parse_age_bins(s: str) -> List[Tuple[float, float, str]]:
    """
    Eingabe z. B. '0-20,21-40,41-60,61-80,81-120,121-999'
    """
    bins = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-")
        lo_f, hi_f = float(lo), float(hi)
        bins.append((lo_f, hi_f, f"{int(lo_f)}-{int(hi_f)}"))
    if not bins:
        raise ValueError("Keine gültigen age_bins angegeben.")
    return bins


def assign_age_class(age_arr: pd.Series, bins: List[Tuple[float, float, str]]) -> pd.Series:
    labels = []
    for age in age_arr.astype(float):
        assigned = "unbekannt"
        for lo, hi, label in bins:
            if lo <= age <= hi:
                assigned = label
                break
        labels.append(assigned)
    return pd.Series(labels, index=age_arr.index)


# -----------------------------------------------------------------------------
# Vorverarbeitung
# -----------------------------------------------------------------------------

def prepare_input(
    df: pd.DataFrame,
    current_year: int,
    col_id: Optional[str],
    col_species: Optional[str],
    col_plant_year: Optional[str],
    col_category: Optional[str],
    fallback_life_years: float,
) -> pd.DataFrame:
    out = df.copy()

    species_col = col_species or pick_first_existing(out, ["species_norm", "BAUMART_L", "species", "Art", "art"])
    plant_col = col_plant_year or pick_first_existing(out, ["PFLANZJAHR_num", "PFLANZJAHR", "plant_year"])
    id_col = col_id or pick_first_existing(out, ["BAUMNUMMER", "id", "ID", "tree_id"])
    cat_col = col_category or pick_first_existing(out, ["KATEGORIE", "category", "standortkategorie"])

    if species_col is None:
        raise ValueError("Keine Artspalte gefunden. Bitte --col_species angeben.")
    if plant_col is None:
        raise ValueError("Keine Pflanzjahr-Spalte gefunden. Bitte --col_plant_year angeben.")

    out["species_sim"] = out[species_col].astype(str).fillna("Unbekannt")
    out["plant_year_sim"] = pd.to_numeric(out[plant_col], errors="coerce")
    out["age0"] = current_year - out["plant_year_sim"]
    out.loc[out["plant_year_sim"].isna(), "age0"] = np.nan

    if id_col is None:
        out["tree_id_sim"] = np.arange(1, len(out) + 1)
    else:
        out["tree_id_sim"] = out[id_col]

    if cat_col is None:
        out["category_sim"] = "unbekannt"
    else:
        out["category_sim"] = out[cat_col].astype(str).fillna("unbekannt")

    if "life_final" in out.columns:
        out["life_expectancy_sim"] = pd.to_numeric(out["life_final"], errors="coerce")
    elif "avg_life_baseline" in out.columns:
        out["life_expectancy_sim"] = pd.to_numeric(out["avg_life_baseline"], errors="coerce")
    else:
        out["life_expectancy_sim"] = np.nan

    out["life_expectancy_sim"] = out["life_expectancy_sim"].fillna(fallback_life_years).clip(lower=5)

    for col, default in [
        ("climate_factor", 1.0),
        ("site_factor", 1.0),
        ("management_factor", 1.0),
        ("urban_factor", 1.0),
    ]:
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    median_age = float(np.nanmedian(out["age0"])) if out["age0"].notna().any() else 15.0
    out["age0"] = out["age0"].fillna(median_age).clip(lower=0)

    out["species_multiplier"] = 1.0
    out["category_multiplier"] = 1.0
    out["replacement_species"] = out["species_sim"]
    out["replacement_life_years"] = np.nan
    out["replacement_climate_factor"] = np.nan
    out["replacement_site_factor"] = np.nan
    out["replacement_management_factor"] = np.nan
    out["replacement_urban_factor"] = np.nan

    return out[[
        "tree_id_sim", "species_sim", "category_sim", "plant_year_sim", "age0",
        "life_expectancy_sim", "climate_factor", "site_factor", "management_factor", "urban_factor",
        "species_multiplier", "category_multiplier", "replacement_species", "replacement_life_years",
        "replacement_climate_factor", "replacement_site_factor", "replacement_management_factor",
        "replacement_urban_factor"
    ]].copy()


def apply_species_parameters(trees: pd.DataFrame, path: Optional[str]) -> Tuple[pd.DataFrame, int]:
    if not path:
        return trees, 0
    tbl = read_table(path).copy()
    species_col = pick_first_existing(tbl, ["species_sim", "species_norm", "species", "Art", "art"])
    if species_col is None:
        raise ValueError("species_params-Datei braucht eine Artspalte, z. B. species_sim oder species_norm.")

    rename_map = {species_col: "species_sim"}
    tbl = tbl.rename(columns=rename_map)

    possible = [
        "species_multiplier",
        "replacement_species",
        "replacement_life_years",
        "replacement_climate_factor",
        "replacement_site_factor",
        "replacement_management_factor",
        "replacement_urban_factor",
    ]
    keep = [c for c in possible if c in tbl.columns] + ["species_sim"]
    tbl = tbl[keep].drop_duplicates(subset=["species_sim"])

    merged = trees.merge(tbl, on="species_sim", how="left", suffixes=("", "_sp"))
    matches = int(merged[[c for c in merged.columns if c.endswith("_sp")]].notna().any(axis=1).sum()) if any(c.endswith("_sp") for c in merged.columns) else 0

    for col in possible:
        sp_col = f"{col}_sp"
        if sp_col in merged.columns:
            if col == "replacement_species":
                merged[col] = merged[sp_col].fillna(merged[col])
            else:
                merged[col] = pd.to_numeric(merged[sp_col], errors="coerce").fillna(merged[col])
            merged = merged.drop(columns=[sp_col])

    return merged, matches


def apply_category_parameters(trees: pd.DataFrame, path: Optional[str]) -> Tuple[pd.DataFrame, int]:
    if not path:
        return trees, 0
    tbl = read_table(path).copy()
    cat_col = pick_first_existing(tbl, ["category_sim", "KATEGORIE", "category", "standortkategorie"])
    if cat_col is None:
        raise ValueError("category_params-Datei braucht eine Kategoriespalte.")
    if "category_multiplier" not in tbl.columns:
        raise ValueError("category_params-Datei braucht die Spalte 'category_multiplier'.")

    tbl = tbl.rename(columns={cat_col: "category_sim"})[["category_sim", "category_multiplier"]].drop_duplicates(subset=["category_sim"])
    merged = trees.merge(tbl, on="category_sim", how="left", suffixes=("", "_cat"))
    matches = int(merged["category_multiplier_cat"].notna().sum()) if "category_multiplier_cat" in merged.columns else 0
    if "category_multiplier_cat" in merged.columns:
        merged["category_multiplier"] = pd.to_numeric(merged["category_multiplier_cat"], errors="coerce").fillna(merged["category_multiplier"])
        merged = merged.drop(columns=["category_multiplier_cat"])
    return merged, matches


# -----------------------------------------------------------------------------
# Modell
# -----------------------------------------------------------------------------

def annual_failure_probability(
    age: np.ndarray,
    life_expectancy: np.ndarray,
    climate_factor: np.ndarray,
    site_factor: np.ndarray,
    management_factor: np.ndarray,
    urban_factor: np.ndarray,
    species_multiplier: np.ndarray,
    category_multiplier: np.ndarray,
    base_rate: float,
    age_midpoint: float,
    age_slope: float,
    age_power: float,
    climate_weight: float,
    site_weight: float,
    management_weight: float,
    urban_weight: float,
    climate_penalty: float,
    year_specific_climate_penalty: float,
    min_p: float,
    max_p: float,
) -> np.ndarray:
    rel_age = age / np.maximum(life_expectancy, 1.0)
    rel_age = np.clip(rel_age, 0.0, 3.0)
    age_term = ((rel_age ** age_power) - age_midpoint) / max(age_slope, 1e-6)

    climate_term = climate_weight * np.maximum(0.0, 1.0 - climate_factor + climate_penalty + year_specific_climate_penalty)
    site_term = site_weight * np.maximum(0.0, 1.0 - site_factor)
    management_term = management_weight * np.maximum(0.0, 1.0 - management_factor)
    urban_term = urban_weight * np.maximum(0.0, 1.0 - urban_factor)

    score = age_term + climate_term + site_term + management_term + urban_term
    p = base_rate * logistic(score)
    p = p * np.maximum(species_multiplier, 0.01) * np.maximum(category_multiplier, 0.01)
    return clamp(p, min_p, max_p)


def climate_penalty_for_year(year_ahead: int, climate_trend_start: float, climate_trend_end: float, years: int) -> float:
    if years <= 1:
        return climate_trend_end
    frac = (year_ahead - 1) / max(years - 1, 1)
    return climate_trend_start + frac * (climate_trend_end - climate_trend_start)


def resolve_replacement_values(
    row: pd.Series,
    default_life_years: float,
    default_climate_factor: float,
    default_site_factor: float,
    default_management_factor: float,
    default_urban_factor: float,
    replacement_same_species: bool,
) -> Dict[str, object]:
    repl_species = row["species_sim"] if replacement_same_species else row.get("replacement_species", row["species_sim"])
    if pd.isna(repl_species) or str(repl_species).strip() == "":
        repl_species = row["species_sim"]

    def choose(col: str, default: float) -> float:
        val = row.get(col, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        return default if np.isnan(val) else val

    return {
        "species_sim": str(repl_species),
        "category_sim": row["category_sim"],
        "life_expectancy_sim": choose("replacement_life_years", default_life_years),
        "climate_factor": choose("replacement_climate_factor", default_climate_factor),
        "site_factor": choose("replacement_site_factor", default_site_factor),
        "management_factor": choose("replacement_management_factor", default_management_factor),
        "urban_factor": choose("replacement_urban_factor", default_urban_factor),
    }


def simulate_one_run(
    trees: pd.DataFrame,
    years: int,
    rng: np.random.Generator,
    age_bins: List[Tuple[float, float, str]],
    base_rate: float,
    age_midpoint: float,
    age_slope: float,
    age_power: float,
    climate_weight: float,
    site_weight: float,
    management_weight: float,
    urban_weight: float,
    climate_trend_start: float,
    climate_trend_end: float,
    extreme_event_interval: int,
    extreme_event_multiplier: float,
    replacement_rate: float,
    replacement_delay: int,
    replacement_same_species: bool,
    replacement_life_years: float,
    replacement_climate_factor: float,
    replacement_site_factor: float,
    replacement_management_factor: float,
    replacement_urban_factor: float,
    min_p: float,
    max_p: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    active = trees.copy().reset_index(drop=True)
    active["alive"] = True
    pending: List[Dict[str, object]] = []

    rows_species = []
    rows_yearly = []

    for y in range(1, years + 1):
        active.loc[active["alive"], "age0"] += 1

        mask_alive = active["alive"].to_numpy()
        alive_idx = active.index[mask_alive]

        climate_penalty = climate_penalty_for_year(y, climate_trend_start, climate_trend_end, years)

        p_fail = annual_failure_probability(
            age=active.loc[mask_alive, "age0"].to_numpy(dtype=float),
            life_expectancy=active.loc[mask_alive, "life_expectancy_sim"].to_numpy(dtype=float),
            climate_factor=active.loc[mask_alive, "climate_factor"].to_numpy(dtype=float),
            site_factor=active.loc[mask_alive, "site_factor"].to_numpy(dtype=float),
            management_factor=active.loc[mask_alive, "management_factor"].to_numpy(dtype=float),
            urban_factor=active.loc[mask_alive, "urban_factor"].to_numpy(dtype=float),
            species_multiplier=active.loc[mask_alive, "species_multiplier"].to_numpy(dtype=float),
            category_multiplier=active.loc[mask_alive, "category_multiplier"].to_numpy(dtype=float),
            base_rate=base_rate,
            age_midpoint=age_midpoint,
            age_slope=age_slope,
            age_power=age_power,
            climate_weight=climate_weight,
            site_weight=site_weight,
            management_weight=management_weight,
            urban_weight=urban_weight,
            climate_penalty=climate_penalty,
            year_specific_climate_penalty=0.0,
            min_p=min_p,
            max_p=max_p,
        )

        extreme_now = extreme_event_interval > 0 and (y % extreme_event_interval == 0)
        if extreme_now:
            p_fail = clamp(p_fail * extreme_event_multiplier, min_p, max_p)

        draws = rng.random(len(p_fail))
        fail_flags = draws < p_fail
        failed_idx = alive_idx[fail_flags]
        failures_this_year = int(len(failed_idx))

        if failures_this_year > 0:
            active.loc[failed_idx, "alive"] = False
            if replacement_rate > 0:
                for idx in failed_idx:
                    if rng.random() <= replacement_rate:
                        repl = resolve_replacement_values(
                            active.loc[idx],
                            default_life_years=replacement_life_years,
                            default_climate_factor=replacement_climate_factor,
                            default_site_factor=replacement_site_factor,
                            default_management_factor=replacement_management_factor,
                            default_urban_factor=replacement_urban_factor,
                            replacement_same_species=replacement_same_species,
                        )
                        pending.append({
                            "years_left": max(0, replacement_delay),
                            **repl,
                        })

        new_pending = []
        new_rows = []
        for item in pending:
            item["years_left"] = int(item["years_left"]) - 1
            if item["years_left"] <= 0:
                new_rows.append({
                    "tree_id_sim": f"new_{y}_{len(new_rows)+1}_{rng.integers(1_000_000)}",
                    "species_sim": item["species_sim"],
                    "category_sim": item["category_sim"],
                    "plant_year_sim": np.nan,
                    "age0": 0.0,
                    "life_expectancy_sim": float(item["life_expectancy_sim"]),
                    "climate_factor": float(item["climate_factor"]),
                    "site_factor": float(item["site_factor"]),
                    "management_factor": float(item["management_factor"]),
                    "urban_factor": float(item["urban_factor"]),
                    "species_multiplier": 1.0,
                    "category_multiplier": 1.0,
                    "replacement_species": item["species_sim"],
                    "replacement_life_years": np.nan,
                    "replacement_climate_factor": np.nan,
                    "replacement_site_factor": np.nan,
                    "replacement_management_factor": np.nan,
                    "replacement_urban_factor": np.nan,
                    "alive": True,
                })
            else:
                new_pending.append(item)
        pending = new_pending
        replacements_this_year = len(new_rows)

        if new_rows:
            active = pd.concat([active, pd.DataFrame(new_rows)], ignore_index=True)

        alive_df = active[active["alive"]].copy()
        alive_df["age_class"] = assign_age_class(alive_df["age0"], age_bins)

        grouped_species = alive_df.groupby(["species_sim", "category_sim"], dropna=False).size().reset_index(name="alive_count")
        grouped_species["year_ahead"] = y
        grouped_species["alive_total"] = int(alive_df.shape[0])
        rows_species.append(grouped_species)

        grouped_age = alive_df.groupby(["category_sim", "age_class"], dropna=False).size().reset_index(name="alive_count")
        grouped_age["year_ahead"] = y

        yearly_total = int(alive_df.shape[0])
        rows_yearly.append(pd.DataFrame({
            "year_ahead": [y],
            "alive_total": [yearly_total],
            "failures_this_year": [failures_this_year],
            "replacements_this_year": [replacements_this_year],
            "net_change_this_year": [replacements_this_year - failures_this_year],
            "extreme_event": [bool(extreme_now)],
            "climate_penalty": [float(climate_penalty)],
        }))
        rows_yearly.append(grouped_age.assign(record_type="age_by_category"))

    species_df = pd.concat(rows_species, ignore_index=True) if rows_species else pd.DataFrame()
    yearly_df = pd.concat(rows_yearly, ignore_index=True) if rows_yearly else pd.DataFrame()
    return species_df, yearly_df


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_species_runs(species_runs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_runs = []
    for i, df in enumerate(species_runs, start=1):
        d = df.copy()
        d["run"] = i
        all_runs.append(d)
    full = pd.concat(all_runs, ignore_index=True)

    rows = []
    for keys, sub in full.groupby(["year_ahead", "species_sim", "category_sim"], dropna=False):
        vals = sub["alive_count"].to_numpy(dtype=float)
        row = {
            "year_ahead": keys[0],
            "species_sim": keys[1],
            "category_sim": keys[2],
            "mean_alive": float(np.mean(vals)),
            "std_alive": float(np.std(vals, ddof=0)),
        }
        row.update(summarize_percentiles(vals))
        rows.append(row)
    species_summary = pd.DataFrame(rows)

    cat_rows = []
    for keys, sub in full.groupby(["year_ahead", "category_sim"], dropna=False):
        vals = sub.groupby("run")["alive_count"].sum().to_numpy(dtype=float)
        row = {
            "year_ahead": keys[0],
            "category_sim": keys[1],
            "mean_alive": float(np.mean(vals)),
            "std_alive": float(np.std(vals, ddof=0)),
        }
        row.update(summarize_percentiles(vals))
        cat_rows.append(row)
    category_summary = pd.DataFrame(cat_rows)

    total_rows = []
    for year, sub in full.groupby("year_ahead"):
        vals = sub.groupby("run")["alive_count"].sum().to_numpy(dtype=float)
        row = {
            "year_ahead": year,
            "mean_alive_total": float(np.mean(vals)),
            "std_alive_total": float(np.std(vals, ddof=0)),
        }
        row.update({f"alive_total_{k}": v for k, v in summarize_percentiles(vals).items()})
        total_rows.append(row)
    total_summary = pd.DataFrame(total_rows)

    return full, species_summary, category_summary, total_summary


def aggregate_yearly_runs(yearly_runs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    annual = []
    agecat = []
    for i, df in enumerate(yearly_runs, start=1):
        if df.empty:
            continue
        d = df.copy()
        d["run"] = i
        annual.append(d[d.columns.intersection([
            "year_ahead", "alive_total", "failures_this_year", "replacements_this_year",
            "net_change_this_year", "extreme_event", "climate_penalty", "run"
        ])].dropna(subset=["year_ahead"]))
        if "record_type" in d.columns:
            agecat.append(d[d["record_type"] == "age_by_category"].copy())

    annual_full = pd.concat(annual, ignore_index=True) if annual else pd.DataFrame()
    agecat_full = pd.concat(agecat, ignore_index=True) if agecat else pd.DataFrame()

    annual_rows = []
    if not annual_full.empty:
        for year, sub in annual_full.groupby("year_ahead"):
            row = {
                "year_ahead": year,
                "mean_alive_total": float(sub["alive_total"].mean()),
                "mean_failures_this_year": float(sub["failures_this_year"].mean()),
                "mean_replacements_this_year": float(sub["replacements_this_year"].mean()),
                "mean_net_change_this_year": float(sub["net_change_this_year"].mean()),
                "extreme_event_share": float(pd.to_numeric(sub["extreme_event"], errors="coerce").fillna(0).mean()),
                "climate_penalty": float(sub["climate_penalty"].mean()),
            }
            annual_rows.append(row)
    annual_summary = pd.DataFrame(annual_rows)

    agecat_rows = []
    if not agecat_full.empty:
        for keys, sub in agecat_full.groupby(["year_ahead", "category_sim", "age_class"], dropna=False):
            vals = pd.to_numeric(sub["alive_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            row = {
                "year_ahead": keys[0],
                "category_sim": keys[1],
                "age_class": keys[2],
                "mean_alive": float(np.mean(vals)),
                "std_alive": float(np.std(vals, ddof=0)),
            }
            row.update(summarize_percentiles(vals))
            agecat_rows.append(row)
    agecat_summary = pd.DataFrame(agecat_rows)

    return annual_summary, agecat_summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stochastisches Stadtbaum-Bestandsmodell v2")
    ap.add_argument("--input", required=True, help="Eingabe-Datei CSV/XLSX")
    ap.add_argument("--out_dir", default="stochastic_output_v2", help="Ausgabeordner")

    ap.add_argument("--col_id", default=None)
    ap.add_argument("--col_species", default=None)
    ap.add_argument("--col_plant_year", default=None)
    ap.add_argument("--col_category", default=None)

    ap.add_argument("--species_params", default=None, help="Optionale CSV/XLSX mit artspezifischen Parametern")
    ap.add_argument("--category_params", default=None, help="Optionale CSV/XLSX mit Kategoriemultiplikatoren")

    ap.add_argument("--current_year", type=int, default=2026)
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--n_runs", type=int, default=500)
    ap.add_argument("--random_seed", type=int, default=42)

    ap.add_argument("--base_rate", type=float, default=0.04)
    ap.add_argument("--age_midpoint", type=float, default=0.90)
    ap.add_argument("--age_slope", type=float, default=0.20)
    ap.add_argument("--age_power", type=float, default=2.0)
    ap.add_argument("--min_p", type=float, default=0.001)
    ap.add_argument("--max_p", type=float, default=0.60)

    ap.add_argument("--climate_weight", type=float, default=2.0)
    ap.add_argument("--site_weight", type=float, default=1.2)
    ap.add_argument("--management_weight", type=float, default=0.6)
    ap.add_argument("--urban_weight", type=float, default=0.8)
    ap.add_argument("--climate_trend_start", type=float, default=0.0, help="zusätzlicher Klimadruck zu Beginn")
    ap.add_argument("--climate_trend_end", type=float, default=0.25, help="zusätzlicher Klimadruck am Ende")

    ap.add_argument("--extreme_event_interval", type=int, default=10, help="alle X Jahre Extremereignis; 0=aus")
    ap.add_argument("--extreme_event_multiplier", type=float, default=1.8)

    ap.add_argument("--replacement_rate", type=float, default=0.8)
    ap.add_argument("--replacement_delay", type=int, default=2)
    ap.add_argument("--replacement_same_species", action="store_true")
    ap.add_argument("--replacement_life_years", type=float, default=80.0)
    ap.add_argument("--replacement_climate_factor", type=float, default=0.95)
    ap.add_argument("--replacement_site_factor", type=float, default=1.0)
    ap.add_argument("--replacement_management_factor", type=float, default=1.0)
    ap.add_argument("--replacement_urban_factor", type=float, default=1.0)

    ap.add_argument("--fallback_life_years", type=float, default=80.0)
    ap.add_argument("--milestones", default="10,20,30,40,50", help="z. B. '10,20,50'")
    ap.add_argument("--age_bins", default="0-20,21-40,41-60,61-80,81-120,121-999")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    age_bins = parse_age_bins(args.age_bins)
    raw = read_table(args.input)
    trees = prepare_input(
        raw,
        current_year=args.current_year,
        col_id=args.col_id,
        col_species=args.col_species,
        col_plant_year=args.col_plant_year,
        col_category=args.col_category,
        fallback_life_years=args.fallback_life_years,
    )

    trees, species_param_matches = apply_species_parameters(trees, args.species_params)
    trees, category_param_matches = apply_category_parameters(trees, args.category_params)

    rng_master = np.random.default_rng(args.random_seed)
    species_runs = []
    yearly_runs = []

    for _ in range(args.n_runs):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        species_df, yearly_df = simulate_one_run(
            trees=trees,
            years=args.years,
            rng=rng,
            age_bins=age_bins,
            base_rate=args.base_rate,
            age_midpoint=args.age_midpoint,
            age_slope=args.age_slope,
            age_power=args.age_power,
            climate_weight=args.climate_weight,
            site_weight=args.site_weight,
            management_weight=args.management_weight,
            urban_weight=args.urban_weight,
            climate_trend_start=args.climate_trend_start,
            climate_trend_end=args.climate_trend_end,
            extreme_event_interval=args.extreme_event_interval,
            extreme_event_multiplier=args.extreme_event_multiplier,
            replacement_rate=args.replacement_rate,
            replacement_delay=args.replacement_delay,
            replacement_same_species=args.replacement_same_species,
            replacement_life_years=args.replacement_life_years,
            replacement_climate_factor=args.replacement_climate_factor,
            replacement_site_factor=args.replacement_site_factor,
            replacement_management_factor=args.replacement_management_factor,
            replacement_urban_factor=args.replacement_urban_factor,
            min_p=args.min_p,
            max_p=args.max_p,
        )
        species_runs.append(species_df)
        yearly_runs.append(yearly_df)

    runs_long, species_summary, category_summary, total_summary = aggregate_species_runs(species_runs)
    annual_summary, agecat_summary = aggregate_yearly_runs(yearly_runs)

    requested_milestones = sorted({int(x.strip()) for x in args.milestones.split(",") if x.strip()})
    milestone_years = sorted({y for y in requested_milestones + [args.years] if y <= args.years})

    species_milestones = species_summary[species_summary["year_ahead"].isin(milestone_years)].copy()
    category_milestones = category_summary[category_summary["year_ahead"].isin(milestone_years)].copy()
    total_milestones = total_summary[total_summary["year_ahead"].isin(milestone_years)].copy()
    annual_milestones = annual_summary[annual_summary["year_ahead"].isin(milestone_years)].copy()
    agecat_milestones = agecat_summary[agecat_summary["year_ahead"].isin(milestone_years)].copy()

    runs_long.to_csv(out_dir / "simulation_runs_long.csv", index=False)
    species_summary.to_csv(out_dir / "species_summary.csv", index=False)
    category_summary.to_csv(out_dir / "category_summary.csv", index=False)
    total_summary.to_csv(out_dir / "total_summary.csv", index=False)
    annual_summary.to_csv(out_dir / "annual_summary.csv", index=False)
    agecat_summary.to_csv(out_dir / "ageclass_category_summary.csv", index=False)

    species_milestones.to_csv(out_dir / "species_milestones.csv", index=False)
    category_milestones.to_csv(out_dir / "category_milestones.csv", index=False)
    total_milestones.to_csv(out_dir / "total_milestones.csv", index=False)
    annual_milestones.to_csv(out_dir / "annual_milestones.csv", index=False)
    agecat_milestones.to_csv(out_dir / "ageclass_category_milestones.csv", index=False)

    with pd.ExcelWriter(out_dir / "stochastic_species_model_results_v2.xlsx", engine="openpyxl") as writer:
        species_summary.to_excel(writer, sheet_name="species_summary", index=False)
        category_summary.to_excel(writer, sheet_name="category_summary", index=False)
        total_summary.to_excel(writer, sheet_name="total_summary", index=False)
        annual_summary.to_excel(writer, sheet_name="annual_summary", index=False)
        agecat_summary.to_excel(writer, sheet_name="ageclass_category", index=False)
        species_milestones.to_excel(writer, sheet_name="species_milestones", index=False)
        category_milestones.to_excel(writer, sheet_name="category_milestones", index=False)
        total_milestones.to_excel(writer, sheet_name="total_milestones", index=False)

    meta = {
        "input": args.input,
        "n_trees_input": len(trees),
        "n_species": int(trees["species_sim"].nunique()),
        "n_categories": int(trees["category_sim"].nunique()),
        "species_params": args.species_params,
        "species_param_matches": species_param_matches,
        "category_params": args.category_params,
        "category_param_matches": category_param_matches,
        "current_year": args.current_year,
        "years": args.years,
        "n_runs": args.n_runs,
        "base_rate": args.base_rate,
        "age_midpoint": args.age_midpoint,
        "age_slope": args.age_slope,
        "age_power": args.age_power,
        "climate_weight": args.climate_weight,
        "site_weight": args.site_weight,
        "management_weight": args.management_weight,
        "urban_weight": args.urban_weight,
        "climate_trend_start": args.climate_trend_start,
        "climate_trend_end": args.climate_trend_end,
        "extreme_event_interval": args.extreme_event_interval,
        "extreme_event_multiplier": args.extreme_event_multiplier,
        "replacement_rate": args.replacement_rate,
        "replacement_delay": args.replacement_delay,
        "replacement_same_species": args.replacement_same_species,
        "replacement_life_years": args.replacement_life_years,
        "replacement_climate_factor": args.replacement_climate_factor,
        "replacement_site_factor": args.replacement_site_factor,
        "replacement_management_factor": args.replacement_management_factor,
        "replacement_urban_factor": args.replacement_urban_factor,
        "milestones": ",".join(map(str, milestone_years)),
        "age_bins": args.age_bins,
    }
    pd.DataFrame({"parameter": list(meta.keys()), "value": list(meta.values())}).to_csv(out_dir / "run_metadata.csv", index=False)

    print("Fertig.")
    print(f"Ausgabeordner: {out_dir.resolve()}")
    print("Dateien:")
    for name in [
        "simulation_runs_long.csv",
        "species_summary.csv",
        "category_summary.csv",
        "total_summary.csv",
        "annual_summary.csv",
        "ageclass_category_summary.csv",
        "species_milestones.csv",
        "category_milestones.csv",
        "total_milestones.csv",
        "annual_milestones.csv",
        "ageclass_category_milestones.csv",
        "stochastic_species_model_results_v2.xlsx",
        "run_metadata.csv",
    ]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
