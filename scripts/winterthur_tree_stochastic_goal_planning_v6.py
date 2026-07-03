#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Winterthur Stadtbaum – stochastisches Ziel- und Szenariomodell v6
=====================================================================

Ziel
----
Dieses Skript simuliert die künftige Entwicklung des aktuellen Baumbestands
für z. B. 25, 50 und 100 Jahre. Im Unterschied zu einer rein heuristischen
Monte-Carlo-Simulation wird die jährliche Ausfallwahrscheinlichkeit aus den
bereits gefällten Bäumen im Baumkataster kalibriert. Zusätzlich können
CityTrees-II, CitiesGOER und TreeGOER direkt integriert werden, sodass
historische Ausfallraten mit artspezifischer Stadteignung, Klimastress,
Standort- und Managementfaktoren kombiniert werden.

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
5) Optional werden für aktive Bäume zusätzliche Faktoren berechnet:
   urban_factor aus CityTrees-II, climate_factor aus CitiesGOER/TreeGOER,
   site_factor aus Standortspalten und management_factor aus Pflegespalten.
6) Die Simulation startet mit den aktiven Bäumen. Jedes Jahr erhält jeder Baum
   abhängig von Alter, Art, Kategorie und diesen optionalen Faktoren eine
   Ausfallwahrscheinlichkeit.
7) Per Zufallszahl wird entschieden, ob ein Baum ausfällt.
8) Optional werden gefällte Bäume nach einer Verzögerung ersetzt.
9) Das Ganze wird mehrfach wiederholt (Monte Carlo). Daraus entstehen Mittelwert
   und Perzentile für 25 / 50 / 100 Jahre.

Wichtig
-------
Das Modell ist eine strategische Szenariosimulation. Es sagt nicht, welcher
Einzelbaum exakt wann ausfällt. Es zeigt erwartete Trends und Bandbreiten für
Bestand, Arten und Kategorien.

Beispielaufruf
--------------
python scripts/winterthur_tree_stochastic_goal_planning_v6.py \
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
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt BOM-Zeichen und Leerzeichen in Spaltennamen."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Liest CSV oder Excel ein. CSV-Separator und Encoding werden robust erkannt."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return clean_columns(pd.read_excel(p))

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return clean_columns(pd.read_csv(p, sep=None, engine="python", encoding=enc))
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"CSV konnte mit keinem Encoding gelesen werden: {encodings}. Letzter Fehler: {last_error}")


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
# Fachliche Baseline-Lebenserwartung pro Baumart
# ---------------------------------------------------------------------------


DEFAULT_BASELINE_LIFE: Dict[str, float] = {
    "Acer campestre": 125, "Acer platanoides": 125, "Acer pseudoplatanus": 125,
    "Aesculus hippocastanum": 105, "Alnus glutinosa": 60, "Amelanchier lamarckii": 60,
    "Betula pendula": 60, "Carpinus betulus": 150, "Castanea sativa": 135,
    "Corylus colurna": 150, "Fagus sylvatica": 155, "Fraxinus excelsior": 120,
    "Ginkgo biloba": 250, "Gleditsia triacanthos": 150, "Juglans regia": 160,
    "Liquidambar styraciflua": 150, "Pinus nigra": 150, "Platanus x acerifolia": 220,
    "Populus tremula": 85, "Prunus avium": 80, "Pyrus communis": 80,
    "Pyrus calleryana": 70, "Quercus robur": 180, "Quercus rubra": 180,
    "Robinia pseudoacacia": 120, "Salix alba": 85, "Sophora japonica": 150,
    "Sorbus intermedia": 120, "Tilia cordata": 180, "Tilia platyphyllos": 180,
    "Tilia europaea": 180, "Ulmus glabra": 130, "Ulmus hollandica": 130,
}


def load_baseline_life(path: Optional[str]) -> Dict[str, float]:
    """Lädt optional eine Arten-Lebensdauer-CSV und überschreibt Default-Werte.

    Erwartete Spalten: species_norm, avg_life_baseline
    """
    d = DEFAULT_BASELINE_LIFE.copy()
    if not path:
        return d
    t = read_table(path)
    required = ["species_norm", "avg_life_baseline"]
    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(f"Baseline-Life CSV braucht Spalten {required}; fehlend: {missing}")
    for _, row in t.iterrows():
        sp = normalize_species_name(row["species_norm"])
        val = safe_float(row["avg_life_baseline"])
        if sp and sp != "unbekannt" and val is not None and val > 0:
            d[sp] = val
    return d


def weighted_choice_indices(rng: np.random.Generator, weights: np.ndarray, size: int) -> np.ndarray:
    """Stabile gewichtete Auswahl; fällt bei ungültigen Gewichten auf Gleichverteilung zurück."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    probs = weights / weights.sum()
    return rng.choice(np.arange(len(weights)), size=size, replace=True, p=probs)


def deterministic_weighted_indices(weights: np.ndarray, size: int) -> np.ndarray:
    """Deterministische Ersatz-Auswahl: nimmt die höchsten Gewichte, zyklisch bei Bedarf."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if len(weights) == 0 or size <= 0:
        return np.array([], dtype=int)
    order = np.argsort(-weights)
    if len(order) >= size:
        return order[:size].astype(int)
    reps = int(np.ceil(size / len(order)))
    return np.tile(order, reps)[:size].astype(int)


# ---------------------------------------------------------------------------
# Externe Faktorberechnung: CityTrees-II, CitiesGOER, TreeGOER, Standort, Pflege
# ---------------------------------------------------------------------------


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def norm_0_1(value: Optional[float], base: float, span: float, mode: str = "high_is_bad") -> float:
    """Normiert einen Klimawert in einen Stressbereich von 0 bis 1."""
    if value is None or span <= 0:
        return 0.0
    if mode == "high_is_bad":
        return max(0.0, min(1.0, (value - base) / span))
    if mode == "low_is_bad":
        return max(0.0, min(1.0, (base - value) / span))
    raise ValueError(f"Unbekannter mode für norm_0_1: {mode}")


def read_citiesgoer_excel(path: str | Path, header_row: int = 6) -> pd.DataFrame:
    return clean_columns(pd.read_excel(path, header=header_row))


def pick_city_row(cities: pd.DataFrame, city_name: str, country_code: str) -> pd.Series:
    required = ["Name", "Country Code"]
    missing = [c for c in required if c not in cities.columns]
    if missing:
        raise ValueError(f"CitiesGOER braucht Spalten {required}; fehlend: {missing}")
    m = (cities["Name"].astype(str).str.lower() == city_name.lower()) & (cities["Country Code"].astype(str) == country_code)
    sub = cities[m]
    if len(sub) == 0:
        sub = cities[cities["Name"].astype(str).str.contains(city_name, case=False, na=False)]
    if len(sub) == 0:
        raise ValueError(f"Stadt nicht gefunden in CitiesGOER: {city_name} ({country_code})")
    if "Population" in sub.columns:
        sub = sub.sort_values("Population", ascending=False)
    return sub.iloc[0]


def read_treegoer_csv(path: str | Path) -> pd.DataFrame:
    t = read_table(path)
    if "species" not in t.columns:
        raise ValueError("TreeGOER CSV braucht eine Spalte 'species'")
    t["species_norm"] = t["species"].apply(normalize_species_name)
    return t


def _bool_text_factor(series: pd.Series, yes_factor: float, no_factor: float) -> pd.Series:
    txt = series.astype(str).str.lower()
    return pd.Series(
        np.where(
            txt.str.contains("ja|yes|true|1", regex=True, na=False),
            yes_factor,
            np.where(txt.str.contains("nein|no|false|0", regex=True, na=False), no_factor, 1.0),
        ),
        index=series.index,
    )


def compute_integrated_factors(active: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Berechnet optionale Einflussfaktoren direkt aus CityTrees/CitiesGOER/TreeGOER.

    Rückgabe:
    - factor_table: pro Baum-ID factor columns und Diagnosewerte
    - factor_diagnostics: kompakte Kennzahlen zur Faktorberechnung
    - climate_components: Einzelwerte des CitiesGOER-Klimastresses
    """
    factor_cols = ["climate_factor", "site_factor", "management_factor", "urban_factor"]
    f = active[[args.col_id, "species_norm", args.col_category]].copy()
    for c in factor_cols:
        f[c] = 1.0

    diagnostics: list[list[object]] = []
    climate_components: list[list[object]] = []
    diagnostics.append(["n_active_for_factor_calc", len(f)])

    # CityTrees-II: urban_factor und Artvulnerabilität
    citytrees_available = False
    if args.citytrees_scores:
        scores = read_table(args.citytrees_scores)
        if "species_latin" not in scores.columns and "species" not in scores.columns:
            raise ValueError("CityTrees-Datei braucht 'species_latin' oder 'species'.")
        species_col_scores = "species_latin" if "species_latin" in scores.columns else "species"
        scores["species_norm"] = scores[species_col_scores].apply(normalize_species_name)
        keep = ["species_norm"] + [c for c in scores.columns if c not in [species_col_scores, "species_norm"]]
        f = f.merge(scores[keep], on="species_norm", how="left")
        citytrees_available = True
        f["urban_overall"] = pd.to_numeric(f.get("urban_overall"), errors="coerce")
        f["urban_factor"] = np.where(f["urban_overall"].notna(), 0.8 + 0.4 * f["urban_overall"], 1.0)
        diagnostics.append(["matched_citytrees_active", int(f["urban_overall"].notna().sum())])
    else:
        diagnostics.append(["matched_citytrees_active", 0])

    # Standort-Faktor
    category_map = parse_mapping(args.category_factor_map)
    f["site_factor_category"] = map_text_multiplier(f[args.col_category], category_map, default=1.0)

    f["site_factor_site_type"] = 1.0
    if args.col_site_type and args.col_site_type in active.columns:
        site_map = parse_mapping(args.site_type_factor_map)
        f["site_factor_site_type"] = map_text_multiplier(active.loc[f.index, args.col_site_type], site_map, default=1.0)

    f["site_factor_sealing"] = 1.0
    if args.col_sealed and args.col_sealed in active.columns:
        f["site_factor_sealing"] = _bool_text_factor(active.loc[f.index, args.col_sealed], args.sealed_yes_factor, args.sealed_no_factor)

    f["site_factor_root_volume"] = 1.0
    if args.col_root_volume and args.col_root_volume in active.columns:
        root_vol = parse_float_series(active.loc[f.index, args.col_root_volume])
        f["site_factor_root_volume"] = np.where(
            root_vol.notna() & (root_vol < args.root_volume_low_threshold),
            args.root_volume_low_factor,
            np.where(root_vol.notna(), args.root_volume_ok_factor, 1.0),
        )

    f["site_factor"] = (
        f["site_factor_category"]
        * f["site_factor_site_type"]
        * f["site_factor_sealing"]
        * f["site_factor_root_volume"]
    )

    # Management-Faktor
    f["management_factor_irrigation"] = 1.0
    if args.col_irrigation and args.col_irrigation in active.columns:
        f["management_factor_irrigation"] = _bool_text_factor(active.loc[f.index, args.col_irrigation], args.irrigation_yes_factor, args.irrigation_no_factor)

    f["management_factor_maintenance"] = 1.0
    if args.col_maintenance and args.col_maintenance in active.columns:
        maintenance_map = parse_mapping(args.maintenance_factor_map)
        f["management_factor_maintenance"] = map_text_multiplier(active.loc[f.index, args.col_maintenance], maintenance_map, default=1.0)

    f["management_factor"] = f["management_factor_irrigation"] * f["management_factor_maintenance"]

    # CitiesGOER lesen, falls Klima oder TreeGOER aktiv
    city_row = None
    if args.citiesgoer_xlsx:
        cities = read_citiesgoer_excel(args.citiesgoer_xlsx, header_row=args.cities_header_row)
        city_row = pick_city_row(cities, args.city_name, args.country_code)
        diagnostics.append(["citiesgoer_city", f"{args.city_name} {args.country_code}"])

    # Proxy-Klima aus CitiesGOER + CityTrees-Vulnerabilität
    f["species_vulnerability_proxy"] = np.nan
    if args.use_proxy_climate:
        if city_row is None:
            raise ValueError("--use_proxy_climate aktiv, aber --citiesgoer_xlsx fehlt.")
        if not citytrees_available:
            raise ValueError("--use_proxy_climate braucht --citytrees_scores für die Artvulnerabilität.")

        bio05 = safe_float(city_row.get("bio05"))
        bio12 = safe_float(city_row.get("bio12"))
        bio06 = safe_float(city_row.get("bio06"))
        bio15 = safe_float(city_row.get("bio15"))
        if bio05 is None or bio12 is None:
            raise ValueError("CitiesGOER enthält bio05/bio12 nicht oder nicht lesbar.")

        heat_stress = norm_0_1(bio05, args.proxy_heat_base, args.proxy_heat_span, mode="high_is_bad")
        drought_stress = norm_0_1(bio12, args.proxy_precip_base, args.proxy_precip_span, mode="low_is_bad")
        winter_stress = norm_0_1(bio06, args.proxy_winter_base, args.proxy_winter_span, mode="low_is_bad")
        variability_stress = norm_0_1(bio15, args.proxy_variability_base, args.proxy_variability_span, mode="high_is_bad")
        weight_sum = args.weight_heat + args.weight_drought + args.weight_winter + args.weight_variability
        if weight_sum <= 0:
            raise ValueError("Summe der Klima-Gewichte muss > 0 sein.")
        city_stress = (
            args.weight_heat * heat_stress
            + args.weight_drought * drought_stress
            + args.weight_winter * winter_stress
            + args.weight_variability * variability_stress
        ) / weight_sum

        for col in ["drought_tol", "low_water_need", "heat_tol", "frost_tol"]:
            f[col] = pd.to_numeric(f.get(col), errors="coerce")
        sw = args.species_weight_drought + args.species_weight_water_need + args.species_weight_heat + args.species_weight_frost
        if sw <= 0:
            raise ValueError("Summe der Art-Gewichte muss > 0 sein.")
        species_vulnerability = (
            args.species_weight_drought * (1 - f["drought_tol"].fillna(0.5))
            + args.species_weight_water_need * (1 - f["low_water_need"].fillna(0.5))
            + args.species_weight_heat * (1 - f["heat_tol"].fillna(0.5))
            + args.species_weight_frost * (1 - f["frost_tol"].fillna(0.5))
        ) / sw
        f["species_vulnerability_proxy"] = species_vulnerability.clip(0.0, 1.0)
        f["city_stress"] = city_stress
        f["city_heat_stress"] = heat_stress
        f["city_drought_stress"] = drought_stress
        f["city_winter_stress"] = winter_stress
        f["city_variability_stress"] = variability_stress
        f["climate_factor"] = np.exp(-args.proxy_scale * city_stress * f["species_vulnerability_proxy"])

        climate_components.extend([
            ["scenario_label", args.scenario_label],
            ["bio05", bio05],
            ["bio12", bio12],
            ["bio06", bio06],
            ["bio15", bio15],
            ["heat_stress", heat_stress],
            ["drought_stress", drought_stress],
            ["winter_stress", winter_stress],
            ["variability_stress", variability_stress],
            ["city_stress", city_stress],
            ["proxy_scale", args.proxy_scale],
            ["climate_mode", "proxy_multifactor"],
        ])

    # Optional TreeGOER überschreibt den Climate-Factor, wenn gesetzt.
    if args.treegoer_csv:
        if city_row is None:
            raise ValueError("TreeGOER aktiv, aber --citiesgoer_xlsx fehlt.")
        tree = read_treegoer_csv(args.treegoer_csv)
        q_col = f"{args.tree_var}_{args.tree_q}"
        if q_col not in tree.columns:
            raise ValueError(f"TreeGOER Spalte nicht gefunden: {q_col}")
        future_val = safe_float(city_row.get(args.future_var_from_cities))
        if future_val is None:
            raise ValueError(f"CitiesGOER future Variable fehlt/nicht lesbar: {args.future_var_from_cities}")
        f = f.merge(tree[["species_norm", q_col]], on="species_norm", how="left", suffixes=("", "_treegoer"))
        f[q_col] = pd.to_numeric(f[q_col], errors="coerce")
        f["treegoer_exceedance"] = np.maximum(0.0, future_val - f[q_col])
        f["climate_factor"] = np.exp(-args.tree_k * f["treegoer_exceedance"])
        f.loc[f[q_col].isna(), "climate_factor"] = 1.0
        diagnostics.append(["matched_treegoer_active", int(f[q_col].notna().sum())])
        climate_components.append(["climate_mode_treegoer_override", f"{q_col}; future={future_val}; k={args.tree_k}"])

    for c in factor_cols:
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(1.0)
        diagnostics.append([f"mean_{c}", round(float(f[c].mean()), 4)])

    return f, pd.DataFrame(diagnostics, columns=["Kennzahl", "Wert"]), pd.DataFrame(climate_components, columns=["Komponente", "Wert"])


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
    annual_new_trees: int,
    annual_new_trees_start_offset: int,
    annual_new_trees_end_offset: Optional[int],
    new_tree_strategy: str,
    target_count: Optional[int],
    deterministic: bool = False,
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
    baseline_life_base = pd.to_numeric(sim_input.get("avg_life_baseline"), errors="coerce").to_numpy(float)
    if len(baseline_life_base) != len(sim_input):
        baseline_life_base = np.full(len(sim_input), np.nan, dtype=float)
    baseline_life_fill = np.nanmedian(baseline_life_base) if np.isfinite(baseline_life_base).any() else 100.0
    baseline_life_base = np.where(np.isfinite(baseline_life_base), baseline_life_base, baseline_life_fill)

    climate_fit_score = 1.0 / np.clip(climate_mult_base, 0.05, 20.0)
    site_fit_score = 1.0 / np.clip(site_mult_base, 0.05, 20.0)
    urban_fit_score = 1.0 / np.clip(urban_mult_base, 0.05, 20.0)
    life_score = np.clip(baseline_life_base, 1.0, None)
    if new_tree_strategy == "same_mix":
        new_tree_weights = np.ones(len(sim_input), dtype=float)
    elif new_tree_strategy == "long_life":
        new_tree_weights = life_score
    elif new_tree_strategy == "climate_fit":
        new_tree_weights = climate_fit_score
    elif new_tree_strategy == "balanced":
        new_tree_weights = life_score * climate_fit_score * site_fit_score * urban_fit_score
    else:
        raise ValueError("new_tree_strategy muss einer dieser Werte sein: same_mix, long_life, climate_fit, balanced")

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

                # Zusätzliche Neupflanzungen unabhängig von Ausfällen
                n_new_planted = 0
                end_offset = annual_new_trees_end_offset if annual_new_trees_end_offset is not None else years
                if annual_new_trees > 0 and annual_new_trees_start_offset <= y <= end_offset:
                    n_new_planted = int(annual_new_trees)
                    pick = deterministic_weighted_indices(new_tree_weights, n_new_planted) if deterministic else weighted_choice_indices(rng, new_tree_weights, n_new_planted)
                    species_idx = np.concatenate([species_idx, species_idx_base[pick]])
                    species_arr = np.concatenate([species_arr, species_arr_base[pick].astype(object)])
                    category_arr = np.concatenate([category_arr, category_arr_base[pick].astype(object)])
                    age = np.concatenate([age, np.zeros(n_new_planted, dtype=int)])
                    alive = np.concatenate([alive, np.ones(n_new_planted, dtype=bool)])
                    climate_mult = np.concatenate([climate_mult, climate_mult_base[pick]])
                    site_mult = np.concatenate([site_mult, site_mult_base[pick]])
                    management_mult = np.concatenate([management_mult, management_mult_base[pick]])
                    urban_mult = np.concatenate([urban_mult, urban_mult_base[pick]])
                    category_mult = np.concatenate([category_mult, category_mult_base[pick]])

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

                if deterministic:
                    n_deaths_expected = int(round(float(np.sum(p_fail))))
                    if n_deaths_expected > 0:
                        severity = p_fail
                        order = np.argsort(-severity)
                        died_idx = idx_alive[order[:min(n_deaths_expected, len(idx_alive))]]
                    else:
                        died_idx = np.array([], dtype=int)
                else:
                    died_flags = rng.random(len(idx_alive)) < p_fail
                    died_idx = idx_alive[died_flags]

                alive[died_idx] = False

                n_deaths = len(died_idx)
                n_repl = 0
                if n_deaths > 0 and replacement_rate > 0:
                    if deterministic:
                        n_repl = int(round(n_deaths * replacement_rate))
                        repl_idx = died_idx[:n_repl]
                    else:
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
                n_new_planted = 0

            n_alive = int(alive.sum())
            total_records.append({
                "run": run,
                "year_offset": y,
                "year": year,
                "n_alive": n_alive,
                "target_reached": bool(target_count is not None and n_alive >= target_count),
            })
            annual_records.append({
                "run": run,
                "year_offset": y,
                "year": year,
                "deaths": int(n_deaths),
                "replacements_scheduled": int(n_repl),
                "additional_new_trees": int(n_new_planted),
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
        additional_new_trees_mean=("additional_new_trees", "mean"),
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

    if "target_reached" in total_long.columns and total_long["target_reached"].any():
        first_hits = (
            total_long[total_long["target_reached"]]
            .sort_values(["run", "year_offset"])
            .groupby("run", as_index=False)
            .first()[["run", "year_offset", "year", "n_alive"]]
        )
        target_summary = pd.DataFrame([
            ["runs_reaching_target", len(first_hits)],
            ["first_year_mean", float(first_hits["year"].mean()) if len(first_hits) else np.nan],
            ["first_year_p50", float(first_hits["year"].median()) if len(first_hits) else np.nan],
            ["first_year_min", int(first_hits["year"].min()) if len(first_hits) else np.nan],
            ["first_year_max", int(first_hits["year"].max()) if len(first_hits) else np.nan],
        ], columns=["Kennzahl", "Wert"])
        out["target_first_hit_by_run"] = first_hits
        out["target_summary"] = target_summary

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
# Konfiguration / Output-Hilfen
# ---------------------------------------------------------------------------


def load_yaml_config(path: Optional[str]) -> Dict[str, object]:
    """Lädt optionale YAML-Konfiguration. Command-line Argumente dürfen überschreiben."""
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("Für --config wird PyYAML benötigt. Installation: conda install pyyaml oder pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Die YAML-Konfiguration muss ein Dictionary/Mapping sein.")
    return data


def normalize_config_value(value):
    """Listenwerte für argparse-kompatible Felder normalisieren."""
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    return value


def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, object]) -> None:
    """Setzt YAML-Werte als Defaults im Parser."""
    if not config:
        return
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(config) - known)
    if unknown:
        print(f"Warnung: Unbekannte Config-Parameter werden ignoriert: {unknown}")
    defaults = {k: normalize_config_value(v) for k, v in config.items() if k in known}
    parser.set_defaults(**defaults)


def make_timestamped_out_dir(base: str, scenario_label: str = "", run_id: Optional[str] = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_label or "scenario")).strip("_")[:60]
    rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("_") if run_id else ""
    name_parts = [stamp]
    if rid:
        name_parts.append(rid)
    if safe_label:
        name_parts.append(safe_label)
    return Path(base) / ("out_" + "_".join(name_parts))


def args_to_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {k: v for k, v in vars(args).items() if k not in {"config"}}


def write_effective_config(out_dir: Path, args: argparse.Namespace) -> None:
    """Speichert alle tatsächlich verwendeten Parameter maschinenlesbar."""
    cfg = args_to_dict(args)
    with open(out_dir / "effective_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
    if yaml is not None:
        with open(out_dir / "effective_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML-Datei mit Modellparametern.")
    pre_args, _ = pre.parse_known_args()
    config = load_yaml_config(pre_args.config)

    ap = argparse.ArgumentParser(
        description="Kalibriertes Monte-Carlo-Modell für Stadtbaum-Bestandsentwicklung.",
        parents=[pre],
    )

    ap.add_argument("--kataster", default=None, help="Baumkataster CSV/XLSX mit aktiven und gefällten Bäumen.")
    ap.add_argument("--planning_csv", default=None, help="Optional: Output aus Planungsmodell v3 mit Faktoren climate/site/management/urban.")
    ap.add_argument("--out_dir", default="output/calibrated_run", help="Output-Ordner. Bei --auto_timestamp_outdir ist dies der Basisordner.")
    ap.add_argument("--auto_timestamp_outdir", action="store_true", help="Output automatisch in out_yyyymmdd_HHMMSS... unter --out_dir speichern.")
    ap.add_argument("--run_id", default=None, help="Optionale Run-ID für Timestamp-Output und Master-Runner.")

    # Direkte Integration von CityTrees-II / CitiesGOER / TreeGOER
    ap.add_argument("--citytrees_scores", default=None, help="Optional: citytrees2_scores.csv für urban_factor und Artvulnerabilität.")
    ap.add_argument("--citiesgoer_xlsx", default=None, help="Optional: CitiesGOER Excel für Zukunftsklima.")
    ap.add_argument("--cities_header_row", type=int, default=6, help="Header-Zeile in CitiesGOER Excel, 0-indexiert.")
    ap.add_argument("--city_name", default="Winterthur")
    ap.add_argument("--country_code", default="CH")
    ap.add_argument("--scenario_label", default="2050s / SSP126")
    ap.add_argument("--use_proxy_climate", action="store_true", help="Aktiviere Klima-Proxy aus CitiesGOER + CityTrees.")
    ap.add_argument("--treegoer_csv", default=None, help="Optional: TreeGOER Species Ranges CSV; überschreibt climate_factor aus Proxy.")
    ap.add_argument("--tree_var", default="bio05")
    ap.add_argument("--tree_q", default="q95")
    ap.add_argument("--tree_k", type=float, default=0.08)
    ap.add_argument("--future_var_from_cities", default="bio05")

    # Spalten
    ap.add_argument("--col_id", default="BAUMNUMMER")
    ap.add_argument("--col_status", default="BAUMSTATUS")
    ap.add_argument("--col_species", default="BAUMART_L")
    ap.add_argument("--col_plant_year", default="PFLANZJAHR")
    ap.add_argument("--col_fall_date", default="FALLDATUM")
    ap.add_argument("--col_category", default="BAUMTYP", help="Kategorie für Auswertung/Risikomultiplikator, z. B. BAUMTYP oder KATEGORIE.")
    ap.add_argument("--col_site_type", default=None)
    ap.add_argument("--col_root_volume", default=None)
    ap.add_argument("--col_sealed", default=None)
    ap.add_argument("--col_irrigation", default=None)
    ap.add_argument("--col_maintenance", default=None)

    # Statuswerte
    ap.add_argument("--active_status", default="Aktiv", help="Kommagetrennte Statuswerte für aktive Bäume.")
    ap.add_argument("--felled_status", default="Gefällt,Gefaellt", help="Kommagetrennte Statuswerte für gefällte Bäume.")

    # Zeitraum / Simulation
    ap.add_argument("--current_year", type=int, default=2026)
    ap.add_argument("--years", type=int, default=100)
    ap.add_argument("--milestones", default="25,50,100")
    ap.add_argument("--n_runs", type=int, default=300)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="Deterministischer Erwartungswertmodus: keine Zufallsziehung pro Baum.")

    # Plausibilitätsfilter / Hazard
    ap.add_argument("--min_age", type=int, default=0)
    ap.add_argument("--max_age", type=int, default=250)
    ap.add_argument("--min_risk_set", type=int, default=20, help="Minimale Anzahl Bäume unter Risiko pro Alter für Rohhazard.")
    ap.add_argument("--smoothing_alpha", type=float, default=0.5, help="Laplace-Smoothing für Hazard.")
    ap.add_argument("--species_shrinkage_strength", type=float, default=80.0, help="Stärke der Glättung kleiner Arten gegen Gattung/global.")
    ap.add_argument("--max_hazard_calibrated", type=float, default=0.35, help="Maximale kalibrierte jährliche Ausfallwahrscheinlichkeit.")

    # Risikomultiplikatoren in der Simulation
    ap.add_argument("--category_multiplier_map", default="strassenbaum=1.15,strasse=1.15,parkbaum=1.00,park=1.00", help="Kategorie-Multiplikatoren auf jährliches Ausfallrisiko.")
    ap.add_argument("--category_factor_map", default="strasse=0.90,straße=0.90,strassenbaum=0.90,park=1.00,parkbaum=1.00", help="Standort-Faktoren auf Lebensdauer; werden in Risikomultiplikatoren umgerechnet.")
    ap.add_argument("--site_type_factor_map", default=None)
    ap.add_argument("--sealed_yes_factor", type=float, default=0.92)
    ap.add_argument("--sealed_no_factor", type=float, default=1.00)
    ap.add_argument("--root_volume_low_threshold", type=float, default=12.0)
    ap.add_argument("--root_volume_low_factor", type=float, default=0.90)
    ap.add_argument("--root_volume_ok_factor", type=float, default=1.00)
    ap.add_argument("--irrigation_yes_factor", type=float, default=1.05)
    ap.add_argument("--irrigation_no_factor", type=float, default=1.00)
    ap.add_argument("--maintenance_factor_map", default="hoch=1.05,mittel=1.00,niedrig=0.95")

    # Proxy-Klima-Parameter
    ap.add_argument("--proxy_heat_base", type=float, default=26.0)
    ap.add_argument("--proxy_heat_span", type=float, default=6.0)
    ap.add_argument("--proxy_precip_base", type=float, default=900.0)
    ap.add_argument("--proxy_precip_span", type=float, default=600.0)
    ap.add_argument("--proxy_winter_base", type=float, default=-8.0)
    ap.add_argument("--proxy_winter_span", type=float, default=8.0)
    ap.add_argument("--proxy_variability_base", type=float, default=60.0)
    ap.add_argument("--proxy_variability_span", type=float, default=40.0)
    ap.add_argument("--proxy_scale", type=float, default=2.5)
    ap.add_argument("--weight_heat", type=float, default=0.40)
    ap.add_argument("--weight_drought", type=float, default=0.35)
    ap.add_argument("--weight_winter", type=float, default=0.15)
    ap.add_argument("--weight_variability", type=float, default=0.10)
    ap.add_argument("--species_weight_drought", type=float, default=0.35)
    ap.add_argument("--species_weight_water_need", type=float, default=0.20)
    ap.add_argument("--species_weight_heat", type=float, default=0.25)
    ap.add_argument("--species_weight_frost", type=float, default=0.20)

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

    # Ersatzpflanzung und Zusatzpflanzung
    ap.add_argument("--replacement_rate", type=float, default=0.8, help="Anteil Ausfälle, die ersetzt werden.")
    ap.add_argument("--replacement_delay", type=int, default=2, help="Jahre zwischen Ausfall und Ersatzpflanzung.")
    ap.add_argument("--replacement_same_species", action="store_true", help="Ersatz mit gleicher Art. Standard ist gleiche Art.")
    ap.add_argument("--annual_new_trees", type=int, default=0, help="Zusätzliche Neupflanzungen pro Jahr unabhängig von Ersatzpflanzungen.")
    ap.add_argument("--annual_new_trees_start_offset", type=int, default=1, help="Erstes Simulationsjahr für zusätzliche Neupflanzungen. 1 = Jahr nach Start.")
    ap.add_argument("--annual_new_trees_end_offset", type=int, default=None, help="Letztes Simulationsjahr für zusätzliche Neupflanzungen. Leer = bis Simulationsende.")
    ap.add_argument("--new_tree_strategy", default="same_mix", choices=["same_mix", "long_life", "climate_fit", "balanced"], help="Auswahlstrategie für zusätzliche Neupflanzungen.")
    ap.add_argument("--baseline_life_csv", default=None, help="Optional: CSV mit species_norm, avg_life_baseline; überschreibt die eingebauten Arten-Lebensdauern.")

    # Zielwert / Monitoring
    ap.add_argument("--target_count", type=int, default=None, help="Optionaler Zielbestand, z. B. 17000. Wird je Lauf ausgewertet.")

    apply_config_defaults(ap, config)
    args = ap.parse_args()

    if args.kataster is None:
        ap.error("--kataster ist erforderlich, entweder als Command-line Argument oder in --config.")

    if args.auto_timestamp_outdir:
        out_dir = make_timestamped_out_dir(args.out_dir, args.scenario_label, args.run_id)
        args.out_dir = str(out_dir)
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_effective_config(out_dir, args)

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

    print("Berechne/übernehme CityTrees-, Klima-, Standort- und Managementfaktoren ...")
    factor_tables = []
    factor_diagnostics_all = []
    climate_components_all = []

    if args.citytrees_scores or args.citiesgoer_xlsx or args.treegoer_csv or args.col_site_type or args.col_root_volume or args.col_sealed or args.col_irrigation or args.col_maintenance:
        external_factors, factor_diag, climate_components = compute_integrated_factors(active, args)
        factor_tables.append(external_factors)
        factor_diagnostics_all.append(factor_diag)
        climate_components_all.append(climate_components)

    if args.planning_csv:
        planning_factors = read_table(args.planning_csv)
        factor_tables.append(planning_factors)
        factor_diagnostics_all.append(pd.DataFrame([["planning_csv_used", args.planning_csv]], columns=["Kennzahl", "Wert"]))

    if factor_tables:
        # Wenn mehrere Faktortabellen vorhanden sind, hat die später angehängte Tabelle Vorrang.
        # Praktisch: planning_csv überschreibt direkt berechnete Faktoren, falls beides gesetzt ist.
        factor_cols = ["climate_factor", "site_factor", "management_factor", "urban_factor"]
        combined = active[[args.col_id]].copy()
        for ft in factor_tables:
            ft = clean_columns(ft.copy())
            if args.col_id not in ft.columns:
                continue
            keep = [args.col_id] + [c for c in factor_cols if c in ft.columns]
            combined = combined.merge(ft[keep], on=args.col_id, how="left", suffixes=("", "_new"))
            for c in factor_cols:
                nc = f"{c}_new"
                if nc in combined.columns:
                    if c not in combined.columns:
                        combined[c] = np.nan
                    combined[c] = pd.to_numeric(combined[nc], errors="coerce").combine_first(pd.to_numeric(combined[c], errors="coerce"))
                    combined = combined.drop(columns=[nc])
        planning_factors = combined
    else:
        planning_factors = None

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

    baseline_life = load_baseline_life(args.baseline_life_csv)
    sim_input["avg_life_baseline"] = sim_input["species_norm"].map(baseline_life)

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
        annual_new_trees=args.annual_new_trees,
        annual_new_trees_start_offset=args.annual_new_trees_start_offset,
        annual_new_trees_end_offset=args.annual_new_trees_end_offset,
        new_tree_strategy=args.new_tree_strategy,
        target_count=args.target_count,
        deterministic=args.deterministic,
    )

    print("Erstelle Auswertungen ...")
    outputs = summarize_runs(total_long, annual_long, species_long, milestones)

    # Speichern CSV
    diagnostics.to_csv(out_dir / "input_diagnostics.csv", index=False, encoding="utf-8")
    hazard_table.to_csv(out_dir / "calibrated_hazard_by_species_age.csv", index=False, encoding="utf-8")
    species_calibration_summary.to_csv(out_dir / "species_calibration_summary.csv", index=False, encoding="utf-8")
    sim_input.to_csv(out_dir / "simulation_input_active_trees.csv", index=False, encoding="utf-8")
    if factor_diagnostics_all:
        pd.concat(factor_diagnostics_all, ignore_index=True).to_csv(out_dir / "factor_diagnostics.csv", index=False, encoding="utf-8")
    if climate_components_all:
        cc = pd.concat([x for x in climate_components_all if not x.empty], ignore_index=True) if any(not x.empty for x in climate_components_all) else pd.DataFrame(columns=["Komponente", "Wert"])
        cc.to_csv(out_dir / "climate_components.csv", index=False, encoding="utf-8")
    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    # Excel Sammeldatei
    xlsx_path = out_dir / "calibrated_monte_carlo_results_v4_integrated.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        diagnostics.to_excel(writer, index=False, sheet_name="input_diagnostics")
        species_calibration_summary.to_excel(writer, index=False, sheet_name="species_calibration")
        if factor_diagnostics_all:
            pd.concat(factor_diagnostics_all, ignore_index=True).to_excel(writer, index=False, sheet_name="factor_diagnostics")
        if climate_components_all:
            cc = pd.concat([x for x in climate_components_all if not x.empty], ignore_index=True) if any(not x.empty for x in climate_components_all) else pd.DataFrame(columns=["Komponente", "Wert"])
            cc.to_excel(writer, index=False, sheet_name="climate_components")
        outputs["total_summary"].to_excel(writer, index=False, sheet_name="total_summary")
        outputs["total_milestones"].to_excel(writer, index=False, sheet_name="total_milestones")
        outputs["annual_summary"].to_excel(writer, index=False, sheet_name="annual_summary")
        if "species_milestones" in outputs:
            outputs["species_milestones"].to_excel(writer, index=False, sheet_name="species_milestones")
        if "species_summary" in outputs:
            outputs["species_summary"].to_excel(writer, index=False, sheet_name="species_summary")
        if "target_summary" in outputs:
            outputs["target_summary"].to_excel(writer, index=False, sheet_name="target_summary")
        if "target_first_hit_by_run" in outputs:
            outputs["target_first_hit_by_run"].to_excel(writer, index=False, sheet_name="target_first_hit")

    # Metadaten
    meta = pd.DataFrame(
        [
            ["kataster", str(args.kataster)],
            ["planning_csv", str(args.planning_csv)],
            ["citytrees_scores", str(args.citytrees_scores)],
            ["citiesgoer_xlsx", str(args.citiesgoer_xlsx)],
            ["treegoer_csv", str(args.treegoer_csv)],
            ["scenario_label", args.scenario_label],
            ["use_proxy_climate", args.use_proxy_climate],
            ["current_year", args.current_year],
            ["years", args.years],
            ["n_runs", args.n_runs],
            ["milestones", ",".join(map(str, milestones))],
            ["replacement_rate", args.replacement_rate],
            ["replacement_delay", args.replacement_delay],
            ["annual_new_trees", args.annual_new_trees],
            ["annual_new_trees_start_offset", args.annual_new_trees_start_offset],
            ["annual_new_trees_end_offset", args.annual_new_trees_end_offset],
            ["new_tree_strategy", args.new_tree_strategy],
            ["baseline_life_csv", args.baseline_life_csv],
            ["target_count", args.target_count],
            ["deterministic", args.deterministic],
            ["auto_timestamp_outdir", args.auto_timestamp_outdir],
            ["run_id", args.run_id],
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
    print(f"- {out_dir / 'calibrated_monte_carlo_results_v4_integrated.xlsx'}")
    print(f"- {out_dir / 'total_milestones.csv'}")
    print(f"- {out_dir / 'species_milestones.csv'}")
    print(f"- {out_dir / 'calibrated_hazard_by_species_age.csv'}")


if __name__ == "__main__":
    main()
