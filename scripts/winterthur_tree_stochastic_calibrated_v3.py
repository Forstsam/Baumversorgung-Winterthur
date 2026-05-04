#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Winterthur Stadtbaum – kalibriertes stochastisches Bestandsmodell v3
=====================================================================

Ziel
----
Dieses Skript simuliert die künftige Entwicklung des aktuellen Baumbestands
für z. B. 25, 50 und 100 Jahre. Im Unterschied zu einer rein heuristischen
Monte-Carlo-Simulation wird die jährliche Ausfallwahrscheinlichkeit aus den
bereits gefällten Bäumen im Baumkataster kalibriert.

Grundidee
---------
1) Aus dem Baumkataster werden aktive und gefällte Bäume getrennt.
2) Für gefällte Bäume wird das Alter bei Fällung berechnet:
       age_at_felling = fall_year - plant_year
3) Für aktive Bäume wird das aktuelle Alter berechnet:
       age_current = current_year - plant_year
4) Aus aktiven und gefällten Bäumen wird eine diskrete altersabhängige Hazard-
   Funktion geschätzt:
       hazard(age) = deaths_at_age / trees_at_risk_at_age
   Aktive Bäume werden dabei als rechtszensierte Beobachtungen behandelt.
5) Die Simulation startet mit den aktiven Bäumen. Jedes Jahr erhält jeder Baum
   abhängig von Alter, Art, Kategorie und optionalen Faktoren eine
   Ausfallwahrscheinlichkeit.
6) Per Zufallszahl wird entschieden, ob ein Baum ausfällt.
7) Optional werden gefällte Bäume nach einer Verzögerung ersetzt.
8) Das Ganze wird mehrfach wiederholt (Monte Carlo). Daraus entstehen Mittelwert
   und Perzentile für 25 / 50 / 100 Jahre.

Wichtig
-------
Das Modell ist eine strategische Szenariosimulation. Es sagt nicht, welcher
Einzelbaum exakt wann ausfällt. Es zeigt erwartete Trends und Bandbreiten für
Bestand, Arten und Kategorien.

Beispielaufruf
--------------
python scripts/winterthur_tree_stochastic_calibrated_v3.py \
  --kataster data/2026-04-13_Baumkataster_gesamte_Daten.csv \
  --out_dir output/calibrated_run_1 \
  --years 100 \
  --n_runs 300 \
  --milestones 25,50,100 \
  --replacement_rate 0.8 \
  --replacement_delay 2
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
Path("C:/Users/samue/Winmodel/stadtbaeume_stochastic_project/data/2026-04-13_Baumkataster_gesamte_Daten.csv").exists()
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt BOM-Zeichen und Leerzeichen in Spaltennamen."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Liest CSV oder Excel ein. CSV-Separator wird automatisch erkannt."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return clean_columns(pd.read_excel(p))
    return clean_columns(pd.read_csv(p, sep=None, engine="python"))


def normalize_species_name(s) -> str:
    """Normalisiert Artnamen auf Genus + species, entfernt Kultivare."""
    if pd.isna(s):
        return "unbekannt"
    s = str(s).strip()
    s = s.replace("×", "x").replace(" X ", " x ")
    s = re.sub(r"\s+'[^']+'", "", s)
    s = re.sub(r'\s+"[^"]+"', "", s)
    tokens = re.split(r"\s+", s)
    if len(tokens) >= 3 and tokens[1].lower() == "x":
        return f"{tokens[0]} x {tokens[2]}".strip()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}".strip()
    return s or "unbekannt"


def parse_year_from_date(series: pd.Series) -> pd.Series:
    """Extrahiert Jahr aus Datumswerten wie 04.12.2018 oder Excel-Datum."""
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt.dt.year


def parse_float_series(series: pd.Series) -> pd.Series:
    """Robuste Umwandlung in Zahlen, auch bei Komma-Dezimaltrennzeichen."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def parse_mapping(text: Optional[str]) -> Dict[str, float]:
    """Format: 'strassenbaum=1.15,parkbaum=1.00'."""
    out: Dict[str, float] = {}
    if not text:
        return out
    for part in str(text).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip().lower()] = float(v.strip())
        except ValueError:
            pass
    return out


def map_text_multiplier(values: pd.Series, mapping: Dict[str, float], default: float = 1.0) -> pd.Series:
    if not mapping:
        return pd.Series(default, index=values.index)

    def pick(v) -> float:
        if pd.isna(v):
            return default
        txt = str(v).strip().lower()
        for key, mult in mapping.items():
            if key in txt:
                return mult
        return default

    return values.apply(pick)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Datenaufbereitung
# ---------------------------------------------------------------------------


def prepare_survival_data(
    df: pd.DataFrame,
    *,
    current_year: int,
    col_status: str,
    col_species: str,
    col_plant_year: str,
    col_fall_date: str,
    col_category: str,
    col_id: str,
    active_status_values: Iterable[str],
    felled_status_values: Iterable[str],
    min_age: int,
    max_age: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bereitet aktive und gefällte Bäume für Kalibrierung und Simulation auf."""
    required = [col_status, col_species, col_plant_year]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Pflichtspalten im Kataster: {missing}")

    if col_id not in df.columns:
        df[col_id] = np.arange(len(df))
    if col_category not in df.columns:
        df[col_category] = "unbekannt"
    if col_fall_date not in df.columns:
        df[col_fall_date] = np.nan

    d = df.copy()
    d["species_norm"] = d[col_species].apply(normalize_species_name)
    d["genus"] = d["species_norm"].str.split().str[0].fillna("unbekannt")
    d["plant_year_num"] = pd.to_numeric(d[col_plant_year], errors="coerce")
    d["fall_year"] = parse_year_from_date(d[col_fall_date])
    d["status_norm"] = d[col_status].astype(str).str.lower().str.strip()

    active_values = {s.lower().strip() for s in active_status_values}
    felled_values = {s.lower().strip() for s in felled_status_values}

    is_active = d["status_norm"].isin(active_values)
    is_felled = d["status_norm"].isin(felled_values)

    active = d[is_active].copy()
    active["age_current"] = current_year - active["plant_year_num"]

    felled = d[is_felled].copy()
    felled["age_at_felling"] = felled["fall_year"] - felled["plant_year_num"]

    active["valid_age"] = active["age_current"].between(min_age, max_age)
    felled["valid_age"] = felled["age_at_felling"].between(min_age, max_age)

    active_valid = active[active["valid_age"] & active["age_current"].notna()].copy()
    felled_valid = felled[felled["valid_age"] & felled["age_at_felling"].notna()].copy()

    diagnostics = pd.DataFrame(
        [
            ["rows_total", len(d)],
            ["rows_active", int(is_active.sum())],
            ["rows_felled", int(is_felled.sum())],
            ["active_valid_age", len(active_valid)],
            ["felled_valid_age", len(felled_valid)],
            ["active_invalid_or_missing_age", len(active) - len(active_valid)],
            ["felled_invalid_or_missing_age", len(felled) - len(felled_valid)],
            ["unique_species_active", active_valid["species_norm"].nunique()],
            ["unique_species_felled", felled_valid["species_norm"].nunique()],
        ],
        columns=["Kennzahl", "Wert"],
    )

    return active_valid, felled_valid, diagnostics


# ---------------------------------------------------------------------------
# Hazard-Kalibrierung
# ---------------------------------------------------------------------------


def _life_table_for_group(active_ages: np.ndarray, death_ages: np.ndarray, max_age: int) -> pd.DataFrame:
    """Diskrete Life Table für eine Gruppe.

    risk_set(age) = Anzahl Bäume, die Alter age erreicht haben.
    deaths(age) = Anzahl Fällungen genau in diesem Alter.
    """
    ages = np.arange(0, max_age + 1)
    active_ages = active_ages[np.isfinite(active_ages)]
    death_ages = death_ages[np.isfinite(death_ages)]

    active_ages = np.clip(np.rint(active_ages).astype(int), 0, max_age)
    death_ages = np.clip(np.rint(death_ages).astype(int), 0, max_age)

    # Bäume unter Risiko: aktive mit aktuellem Alter >= age + gefällte mit Fällalter >= age
    risk_active = np.array([(active_ages >= a).sum() for a in ages], dtype=float)
    risk_felled = np.array([(death_ages >= a).sum() for a in ages], dtype=float)
    risk_set = risk_active + risk_felled
    deaths = np.bincount(death_ages, minlength=max_age + 1).astype(float)

    return pd.DataFrame({"age": ages, "risk_set": risk_set, "deaths": deaths})


def calibrate_hazards(
    active: pd.DataFrame,
    felled: pd.DataFrame,
    *,
    max_age: int,
    min_risk_set: int,
    smoothing_alpha: float,
    species_shrinkage_strength: float,
    max_hazard_calibrated: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kalibriert Hazard-Kurven global, pro Gattung und pro Art.

    Die Artkurve wird gegen die Gattungskurve und globale Kurve geschrumpft,
    damit kleine Artenbestände nicht extreme Zufallswerte erzeugen.
    """
    # Global
    global_lt = _life_table_for_group(
        active["age_current"].to_numpy(float),
        felled["age_at_felling"].to_numpy(float),
        max_age,
    )
    global_lt["hazard_raw"] = (global_lt["deaths"] + smoothing_alpha) / (
        global_lt["risk_set"] + 2 * smoothing_alpha
    )
    global_lt.loc[global_lt["risk_set"] < min_risk_set, "hazard_raw"] = np.nan
    global_lt["hazard_global"] = global_lt["hazard_raw"].interpolate(limit_direction="both").fillna(0.002)
    global_lt["hazard_global"] = global_lt["hazard_global"].clip(0.0001, max_hazard_calibrated)

    records = []
    summaries = []

    # Genus tables
    genus_tables: Dict[str, pd.DataFrame] = {}
    for genus in sorted(set(active["genus"]).union(set(felled["genus"]))):
        a = active.loc[active["genus"] == genus, "age_current"].to_numpy(float)
        f = felled.loc[felled["genus"] == genus, "age_at_felling"].to_numpy(float)
        lt = _life_table_for_group(a, f, max_age)
        lt["hazard_raw"] = (lt["deaths"] + smoothing_alpha) / (lt["risk_set"] + 2 * smoothing_alpha)
        lt.loc[lt["risk_set"] < min_risk_set, "hazard_raw"] = np.nan
        lt = lt.merge(global_lt[["age", "hazard_global"]], on="age", how="left")
        # Genus-Shrinkage gegen global
        w = lt["risk_set"] / (lt["risk_set"] + species_shrinkage_strength)
        lt["hazard_genus"] = (w * lt["hazard_raw"].fillna(lt["hazard_global"]) + (1 - w) * lt["hazard_global"])
        lt["hazard_genus"] = lt["hazard_genus"].interpolate(limit_direction="both").fillna(lt["hazard_global"])
        lt["hazard_genus"] = lt["hazard_genus"].clip(0.0001, max_hazard_calibrated)
        genus_tables[genus] = lt[["age", "hazard_genus"]]

    # Species tables
    species_all = sorted(set(active["species_norm"]).union(set(felled["species_norm"])))
    for species in species_all:
        genus = str(species).split()[0] if species else "unbekannt"
        a = active.loc[active["species_norm"] == species, "age_current"].to_numpy(float)
        f = felled.loc[felled["species_norm"] == species, "age_at_felling"].to_numpy(float)
        lt = _life_table_for_group(a, f, max_age)
        lt["hazard_raw"] = (lt["deaths"] + smoothing_alpha) / (lt["risk_set"] + 2 * smoothing_alpha)
        lt.loc[lt["risk_set"] < min_risk_set, "hazard_raw"] = np.nan
        lt = lt.merge(global_lt[["age", "hazard_global"]], on="age", how="left")
        if genus in genus_tables:
            lt = lt.merge(genus_tables[genus], on="age", how="left")
        else:
            lt["hazard_genus"] = lt["hazard_global"]

        # Art wird gegen Gattung geschrumpft. Danach liegt immer eine komplette Kurve vor.
        w = lt["risk_set"] / (lt["risk_set"] + species_shrinkage_strength)
        lt["hazard_calibrated"] = (
            w * lt["hazard_raw"].fillna(lt["hazard_genus"])
            + (1 - w) * lt["hazard_genus"]
        )
        lt["hazard_calibrated"] = lt["hazard_calibrated"].interpolate(limit_direction="both")
        lt["hazard_calibrated"] = lt["hazard_calibrated"].fillna(lt["hazard_global"])
        lt["hazard_calibrated"] = lt["hazard_calibrated"].clip(0.0001, max_hazard_calibrated)

        tmp = lt[["age", "risk_set", "deaths", "hazard_raw", "hazard_global", "hazard_genus", "hazard_calibrated"]].copy()
        tmp.insert(0, "species_norm", species)
        records.append(tmp)

        summaries.append(
            {
                "species_norm": species,
                "genus": genus,
                "n_active": int(len(a)),
                "n_felled": int(len(f)),
                "median_age_active": float(np.nanmedian(a)) if len(a) else np.nan,
                "median_age_felled": float(np.nanmedian(f)) if len(f) else np.nan,
                "mean_age_felled": float(np.nanmean(f)) if len(f) else np.nan,
                "max_age_felled": float(np.nanmax(f)) if len(f) else np.nan,
            }
        )

    hazard_table = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return hazard_table, summary


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def build_hazard_lookup(hazard_table: pd.DataFrame, max_age: int) -> Tuple[Dict[str, int], np.ndarray]:
    species = sorted(hazard_table["species_norm"].dropna().unique())
    species_to_idx = {sp: i for i, sp in enumerate(species)}
    lookup = np.full((len(species), max_age + 1), 0.002, dtype=float)
    for sp, idx in species_to_idx.items():
        sub = hazard_table[hazard_table["species_norm"] == sp].sort_values("age")
        vals = sub["hazard_calibrated"].to_numpy(float)
        if len(vals) >= max_age + 1:
            lookup[idx, :] = vals[: max_age + 1]
        else:
            lookup[idx, : len(vals)] = vals
            lookup[idx, len(vals) :] = vals[-1] if len(vals) else 0.002
    return species_to_idx, lookup


def prepare_active_for_simulation(
    active: pd.DataFrame,
    *,
    col_id: str,
    col_category: str,
    category_multiplier_map: Dict[str, float],
    planning_factors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Erstellt Simulations-Input für aktive Bäume.

    Falls planning_factors vorhanden ist, werden climate/site/management/urban_factor übernommen.
    Sonst werden neutrale Faktoren 1.0 verwendet.
    """
    sim = active.copy()

    factor_cols = ["climate_factor", "site_factor", "management_factor", "urban_factor"]
    for c in factor_cols:
        sim[c] = 1.0

    if planning_factors is not None and not planning_factors.empty:
        pf = planning_factors.copy()
        pf = clean_columns(pf)
        if col_id in pf.columns:
            keep_cols = [col_id] + [c for c in factor_cols if c in pf.columns]
            sim = sim.merge(pf[keep_cols], on=col_id, how="left", suffixes=("", "_plan"))
            for c in factor_cols:
                pc = f"{c}_plan"
                if pc in sim.columns:
                    sim[c] = pd.to_numeric(sim[pc], errors="coerce").fillna(sim[c])
                    sim = sim.drop(columns=[pc])

    # Kategoriemultiplikator direkt auf Ausfallrisiko, z. B. Strassenbaum = 1.15
    sim["category_multiplier"] = map_text_multiplier(sim[col_category], category_multiplier_map, default=1.0)

    # Faktoren für Risiko: niedriger climate_factor bedeutet höheres Risiko.
    # Umrechnung: climate_factor 0.8 -> risk climate multiplier > 1.
    sim["climate_risk_multiplier"] = 1.0 / pd.to_numeric(sim["climate_factor"], errors="coerce").fillna(1.0).clip(0.2, 2.0)
    sim["site_risk_multiplier"] = 1.0 / pd.to_numeric(sim["site_factor"], errors="coerce").fillna(1.0).clip(0.2, 2.0)
    sim["management_risk_multiplier"] = 1.0 / pd.to_numeric(sim["management_factor"], errors="coerce").fillna(1.0).clip(0.2, 2.0)
    sim["urban_risk_multiplier"] = 1.0 / pd.to_numeric(sim["urban_factor"], errors="coerce").fillna(1.0).clip(0.2, 2.0)

    return sim


def run_monte_carlo(
    sim_input: pd.DataFrame,
    hazard_table: pd.DataFrame,
    *,
    years: int,
    n_runs: int,
    current_year: int,
    max_age: int,
    random_seed: int,
    climate_trend_start: float,
    climate_trend_end: float,
    climate_weight: float,
    site_weight: float,
    management_weight: float,
    urban_weight: float,
    category_weight: float,
    min_p: float,
    max_p: float,
    extreme_event_interval: int,
    extreme_event_multiplier: float,
    replacement_rate: float,
    replacement_delay: int,
    replacement_same_species: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Führt Monte-Carlo-Simulation durch."""
    rng_master = np.random.default_rng(random_seed)
    species_to_idx, hazard_lookup = build_hazard_lookup(hazard_table, max_age)
    default_species_idx = 0

    # Initiale Arrays
    species_arr_base = sim_input["species_norm"].astype(str).to_numpy()
    species_idx_base = np.array([species_to_idx.get(sp, default_species_idx) for sp in species_arr_base], dtype=int)
    category_arr_base = sim_input["KATEGORIE_SIM"].astype(str).to_numpy()
    age_base = np.rint(sim_input["age_current"].to_numpy(float)).astype(int)
    age_base = np.clip(age_base, 0, max_age)

    climate_mult_base = sim_input["climate_risk_multiplier"].to_numpy(float)
    site_mult_base = sim_input["site_risk_multiplier"].to_numpy(float)
    management_mult_base = sim_input["management_risk_multiplier"].to_numpy(float)
    urban_mult_base = sim_input["urban_risk_multiplier"].to_numpy(float)
    category_mult_base = sim_input["category_multiplier"].to_numpy(float)

    total_records = []
    annual_records = []
    species_records = []

    for run in range(n_runs):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        species_idx = species_idx_base.copy()
        species_arr = species_arr_base.copy()
        category_arr = category_arr_base.copy()
        age = age_base.copy()
        alive = np.ones(len(age), dtype=bool)

        climate_mult = climate_mult_base.copy()
        site_mult = site_mult_base.copy()
        management_mult = management_mult_base.copy()
        urban_mult = urban_mult_base.copy()
        category_mult = category_mult_base.copy()

        # Ersatzwarteschlange: Liste von Dicts mit due_year und Anzahl/Attribute
        replacement_queue: list[dict] = []

        for y in range(0, years + 1):
            year = current_year + y

            if y > 0:
                # Geplante Ersatzpflanzungen hinzufügen
                due = [item for item in replacement_queue if item["due_year"] == year]
                if due:
                    for item in due:
                        n_new = item["n"]
                        if n_new <= 0:
                            continue
                        sp = item["species"]
                        cat = item["category"]
                        sp_idx = species_to_idx.get(sp, default_species_idx)
                        species_idx = np.concatenate([species_idx, np.full(n_new, sp_idx, dtype=int)])
                        species_arr = np.concatenate([species_arr, np.full(n_new, sp, dtype=object)])
                        category_arr = np.concatenate([category_arr, np.full(n_new, cat, dtype=object)])
                        age = np.concatenate([age, np.zeros(n_new, dtype=int)])
                        alive = np.concatenate([alive, np.ones(n_new, dtype=bool)])
                        climate_mult = np.concatenate([climate_mult, np.full(n_new, item["climate_mult"])])
                        site_mult = np.concatenate([site_mult, np.full(n_new, item["site_mult"])])
                        management_mult = np.concatenate([management_mult, np.full(n_new, item["management_mult"])])
                        urban_mult = np.concatenate([urban_mult, np.full(n_new, item["urban_mult"])])
                        category_mult = np.concatenate([category_mult, np.full(n_new, item["category_mult"])])
                    replacement_queue = [item for item in replacement_queue if item["due_year"] != year]

                # Alter erhöhen
                age[alive] += 1
                age = np.clip(age, 0, max_age)

                # Hazard ziehen und modifizieren
                idx_alive = np.where(alive)[0]
                base_p = hazard_lookup[species_idx[idx_alive], age[idx_alive]]

                # Klimatrend: zusätzliches Risiko nimmt linear zu.
                if years > 0:
                    trend = climate_trend_start + (climate_trend_end - climate_trend_start) * (y / years)
                else:
                    trend = climate_trend_start

                risk_mult = (
                    np.power(climate_mult[idx_alive], climate_weight) * (1.0 + trend)
                    * np.power(site_mult[idx_alive], site_weight)
                    * np.power(management_mult[idx_alive], management_weight)
                    * np.power(urban_mult[idx_alive], urban_weight)
                    * np.power(category_mult[idx_alive], category_weight)
                )

                p_fail = np.clip(base_p * risk_mult, min_p, max_p)

                # Extremereignis: alle x Jahre erhöhtes Risiko
                if extreme_event_interval > 0 and y % extreme_event_interval == 0:
                    p_fail = np.clip(p_fail * extreme_event_multiplier, min_p, max_p)

                died_flags = rng.random(len(idx_alive)) < p_fail
                died_idx = idx_alive[died_flags]
                alive[died_idx] = False

                n_deaths = len(died_idx)
                n_repl = 0
                if n_deaths > 0 and replacement_rate > 0:
                    replace_flags = rng.random(n_deaths) < replacement_rate
                    repl_idx = died_idx[replace_flags]
                    n_repl = len(repl_idx)
                    if n_repl > 0:
                        # Gruppiert nach Art und Kategorie, damit die Queue klein bleibt.
                        repl_df = pd.DataFrame({
                            "species": species_arr[repl_idx] if replacement_same_species else species_arr[repl_idx],
                            "category": category_arr[repl_idx],
                            "climate_mult": climate_mult[repl_idx],
                            "site_mult": site_mult[repl_idx],
                            "management_mult": management_mult[repl_idx],
                            "urban_mult": urban_mult[repl_idx],
                            "category_mult": category_mult[repl_idx],
                        })
                        grouped = repl_df.groupby(
                            ["species", "category", "climate_mult", "site_mult", "management_mult", "urban_mult", "category_mult"],
                            dropna=False,
                        ).size().reset_index(name="n")
                        due_year = year + replacement_delay
                        for _, row in grouped.iterrows():
                            replacement_queue.append({
                                "due_year": due_year,
                                "species": str(row["species"]),
                                "category": str(row["category"]),
                                "climate_mult": float(row["climate_mult"]),
                                "site_mult": float(row["site_mult"]),
                                "management_mult": float(row["management_mult"]),
                                "urban_mult": float(row["urban_mult"]),
                                "category_mult": float(row["category_mult"]),
                                "n": int(row["n"]),
                            })
            else:
                n_deaths = 0
                n_repl = 0

            n_alive = int(alive.sum())
            total_records.append({"run": run, "year_offset": y, "year": year, "n_alive": n_alive})
            annual_records.append({
                "run": run,
                "year_offset": y,
                "year": year,
                "deaths": int(n_deaths),
                "replacements_scheduled": int(n_repl),
                "n_alive": n_alive,
            })

            # Nach Arten aggregieren; nur lebende Bäume.
            if y in {0, 25, 50, 100, years} or y % 10 == 0:
                alive_species = pd.Series(species_arr[alive], name="species_norm")
                vc = alive_species.value_counts().reset_index()
                vc.columns = ["species_norm", "n_alive"]
                vc["run"] = run
                vc["year_offset"] = y
                vc["year"] = year
                species_records.append(vc)

    return (
        pd.DataFrame(total_records),
        pd.DataFrame(annual_records),
        pd.concat(species_records, ignore_index=True) if species_records else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------


def summarize_runs(total_long: pd.DataFrame, annual_long: pd.DataFrame, species_long: pd.DataFrame, milestones: list[int]) -> Dict[str, pd.DataFrame]:
    def agg_numeric(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
        return df.groupby(group_cols)[value_col].agg(
            mean="mean",
            p05=lambda x: np.percentile(x, 5),
            p25=lambda x: np.percentile(x, 25),
            p50="median",
            p75=lambda x: np.percentile(x, 75),
            p95=lambda x: np.percentile(x, 95),
            min="min",
            max="max",
        ).reset_index()

    total_summary = agg_numeric(total_long, ["year_offset", "year"], "n_alive")
    annual_summary = annual_long.groupby(["year_offset", "year"]).agg(
        deaths_mean=("deaths", "mean"),
        deaths_p50=("deaths", "median"),
        deaths_p95=("deaths", lambda x: np.percentile(x, 95)),
        replacements_scheduled_mean=("replacements_scheduled", "mean"),
        n_alive_mean=("n_alive", "mean"),
    ).reset_index()

    total_milestones = total_summary[total_summary["year_offset"].isin(milestones)].copy()
    annual_milestones = annual_summary[annual_summary["year_offset"].isin(milestones)].copy()

    out = {
        "total_runs_long": total_long,
        "annual_runs_long": annual_long,
        "total_summary": total_summary,
        "annual_summary": annual_summary,
        "total_milestones": total_milestones,
        "annual_milestones": annual_milestones,
    }

    if not species_long.empty:
        species_summary = species_long.groupby(["species_norm", "year_offset", "year"])["n_alive"].agg(
            mean="mean",
            p05=lambda x: np.percentile(x, 5),
            p50="median",
            p95=lambda x: np.percentile(x, 95),
        ).reset_index()
        species_milestones = species_summary[species_summary["year_offset"].isin(milestones)].copy()
        out["species_runs_long"] = species_long
        out["species_summary"] = species_summary
        out["species_milestones"] = species_milestones

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Kalibriertes Monte-Carlo-Modell für Stadtbaum-Bestandsentwicklung.")

    ap.add_argument("--kataster", required=True, help="Baumkataster CSV/XLSX mit aktiven und gefällten Bäumen.")
    ap.add_argument("--planning_csv", default=None, help="Optional: Output aus Planungsmodell v3 mit Faktoren climate/site/management/urban.")
    ap.add_argument("--out_dir", default="output/calibrated_run", help="Output-Ordner.")

    # Spalten
    ap.add_argument("--col_id", default="BAUMNUMMER")
    ap.add_argument("--col_status", default="BAUMSTATUS")
    ap.add_argument("--col_species", default="BAUMART_L")
    ap.add_argument("--col_plant_year", default="PFLANZJAHR")
    ap.add_argument("--col_fall_date", default="FALLDATUM")
    ap.add_argument("--col_category", default="BAUMTYP", help="Kategorie für Auswertung/Risikomultiplikator, z. B. BAUMTYP oder KATEGORIE.")

    # Statuswerte
    ap.add_argument("--active_status", default="Aktiv", help="Kommagetrennte Statuswerte für aktive Bäume.")
    ap.add_argument("--felled_status", default="Gefällt,Gefaellt", help="Kommagetrennte Statuswerte für gefällte Bäume.")

    # Zeitraum / Simulation
    ap.add_argument("--current_year", type=int, default=2026)
    ap.add_argument("--years", type=int, default=100)
    ap.add_argument("--milestones", default="25,50,100")
    ap.add_argument("--n_runs", type=int, default=300)
    ap.add_argument("--random_seed", type=int, default=42)

    # Plausibilitätsfilter / Hazard
    ap.add_argument("--min_age", type=int, default=0)
    ap.add_argument("--max_age", type=int, default=250)
    ap.add_argument("--min_risk_set", type=int, default=20, help="Minimale Anzahl Bäume unter Risiko pro Alter für Rohhazard.")
    ap.add_argument("--smoothing_alpha", type=float, default=0.5, help="Laplace-Smoothing für Hazard.")
    ap.add_argument("--species_shrinkage_strength", type=float, default=80.0, help="Stärke der Glättung kleiner Arten gegen Gattung/global.")
    ap.add_argument("--max_hazard_calibrated", type=float, default=0.35, help="Maximale kalibrierte jährliche Ausfallwahrscheinlichkeit.")

    # Risikomultiplikatoren in der Simulation
    ap.add_argument("--category_multiplier_map", default="strassenbaum=1.15,strasse=1.15,parkbaum=1.00,park=1.00", help="Kategorie-Multiplikatoren.")
    ap.add_argument("--climate_trend_start", type=float, default=0.0)
    ap.add_argument("--climate_trend_end", type=float, default=0.25)
    ap.add_argument("--climate_weight", type=float, default=1.0)
    ap.add_argument("--site_weight", type=float, default=1.0)
    ap.add_argument("--management_weight", type=float, default=0.6)
    ap.add_argument("--urban_weight", type=float, default=0.8)
    ap.add_argument("--category_weight", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.0001)
    ap.add_argument("--max_p", type=float, default=0.50)

    # Extremereignisse
    ap.add_argument("--extreme_event_interval", type=int, default=0, help="0 = keine Extremereignisse; sonst alle n Jahre.")
    ap.add_argument("--extreme_event_multiplier", type=float, default=1.8)

    # Ersatzpflanzung
    ap.add_argument("--replacement_rate", type=float, default=0.8, help="Anteil Ausfälle, die ersetzt werden.")
    ap.add_argument("--replacement_delay", type=int, default=2, help="Jahre zwischen Ausfall und Ersatzpflanzung.")
    ap.add_argument("--replacement_same_species", action="store_true", help="Ersatz mit gleicher Art. Standard ist ebenfalls gleiche Art, Platzhalter für spätere Erweiterung.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_status_values = [x.strip() for x in args.active_status.split(",") if x.strip()]
    felled_status_values = [x.strip() for x in args.felled_status.split(",") if x.strip()]
    milestones = parse_int_list(args.milestones)
    if 0 not in milestones:
        milestones = [0] + milestones
    if args.years not in milestones:
        milestones.append(args.years)
    milestones = sorted(set([m for m in milestones if 0 <= m <= args.years]))

    print("Lese Kataster ...")
    raw = read_table(args.kataster)

    active, felled, diagnostics = prepare_survival_data(
        raw,
        current_year=args.current_year,
        col_status=args.col_status,
        col_species=args.col_species,
        col_plant_year=args.col_plant_year,
        col_fall_date=args.col_fall_date,
        col_category=args.col_category,
        col_id=args.col_id,
        active_status_values=active_status_values,
        felled_status_values=felled_status_values,
        min_age=args.min_age,
        max_age=args.max_age,
    )

    print(f"Aktive Bäume gültig: {len(active)}")
    print(f"Gefällte Bäume gültig: {len(felled)}")

    print("Kalibriere Hazard-Kurven aus Fälldaten ...")
    hazard_table, species_calibration_summary = calibrate_hazards(
        active,
        felled,
        max_age=args.max_age,
        min_risk_set=args.min_risk_set,
        smoothing_alpha=args.smoothing_alpha,
        species_shrinkage_strength=args.species_shrinkage_strength,
        max_hazard_calibrated=args.max_hazard_calibrated,
    )

    planning_factors = read_table(args.planning_csv) if args.planning_csv else None
    active_for_sim = active.copy()
    active_for_sim["KATEGORIE_SIM"] = active_for_sim[args.col_category].fillna("unbekannt").astype(str)
    category_map = parse_mapping(args.category_multiplier_map)
    sim_input = prepare_active_for_simulation(
        active_for_sim,
        col_id=args.col_id,
        col_category=args.col_category,
        category_multiplier_map=category_map,
        planning_factors=planning_factors,
    )
    # ensure KATEGORIE_SIM survives after optional merge
    sim_input["KATEGORIE_SIM"] = active_for_sim["KATEGORIE_SIM"].to_numpy()

    print(f"Starte Monte-Carlo-Simulation: {args.n_runs} Läufe, {args.years} Jahre ...")
    total_long, annual_long, species_long = run_monte_carlo(
        sim_input,
        hazard_table,
        years=args.years,
        n_runs=args.n_runs,
        current_year=args.current_year,
        max_age=args.max_age,
        random_seed=args.random_seed,
        climate_trend_start=args.climate_trend_start,
        climate_trend_end=args.climate_trend_end,
        climate_weight=args.climate_weight,
        site_weight=args.site_weight,
        management_weight=args.management_weight,
        urban_weight=args.urban_weight,
        category_weight=args.category_weight,
        min_p=args.min_p,
        max_p=args.max_p,
        extreme_event_interval=args.extreme_event_interval,
        extreme_event_multiplier=args.extreme_event_multiplier,
        replacement_rate=args.replacement_rate,
        replacement_delay=args.replacement_delay,
        replacement_same_species=True,
    )

    print("Erstelle Auswertungen ...")
    outputs = summarize_runs(total_long, annual_long, species_long, milestones)

    # Speichern CSV
    diagnostics.to_csv(out_dir / "input_diagnostics.csv", index=False, encoding="utf-8")
    hazard_table.to_csv(out_dir / "calibrated_hazard_by_species_age.csv", index=False, encoding="utf-8")
    species_calibration_summary.to_csv(out_dir / "species_calibration_summary.csv", index=False, encoding="utf-8")
    sim_input.to_csv(out_dir / "simulation_input_active_trees.csv", index=False, encoding="utf-8")
    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    # Excel Sammeldatei
    xlsx_path = out_dir / "calibrated_monte_carlo_results_v3.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        diagnostics.to_excel(writer, index=False, sheet_name="input_diagnostics")
        species_calibration_summary.to_excel(writer, index=False, sheet_name="species_calibration")
        outputs["total_summary"].to_excel(writer, index=False, sheet_name="total_summary")
        outputs["total_milestones"].to_excel(writer, index=False, sheet_name="total_milestones")
        outputs["annual_summary"].to_excel(writer, index=False, sheet_name="annual_summary")
        if "species_milestones" in outputs:
            outputs["species_milestones"].to_excel(writer, index=False, sheet_name="species_milestones")
        if "species_summary" in outputs:
            outputs["species_summary"].to_excel(writer, index=False, sheet_name="species_summary")

    # Metadaten
    meta = pd.DataFrame(
        [
            ["kataster", str(args.kataster)],
            ["planning_csv", str(args.planning_csv)],
            ["current_year", args.current_year],
            ["years", args.years],
            ["n_runs", args.n_runs],
            ["milestones", ",".join(map(str, milestones))],
            ["replacement_rate", args.replacement_rate],
            ["replacement_delay", args.replacement_delay],
            ["climate_trend_end", args.climate_trend_end],
            ["category_multiplier_map", args.category_multiplier_map],
            ["min_risk_set", args.min_risk_set],
            ["species_shrinkage_strength", args.species_shrinkage_strength],
        ],
        columns=["Parameter", "Wert"],
    )
    meta.to_csv(out_dir / "run_metadata.csv", index=False, encoding="utf-8")

    print("Fertig.")
    print(f"Ausgabeordner: {out_dir.resolve()}")
    print("Wichtige Dateien:")
    print(f"- {out_dir / 'calibrated_monte_carlo_results_v3.xlsx'}")
    print(f"- {out_dir / 'total_milestones.csv'}")
    print(f"- {out_dir / 'species_milestones.csv'}")
    print(f"- {out_dir / 'calibrated_hazard_by_species_age.csv'}")


if __name__ == "__main__":
    main()
