#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Winterthur Stadtbaum – Nachpflanzungsplanung (erweiterte Klima-Version)
------------------------------------------------------------------------
Dieses Skript erweitert die ursprüngliche Version um:
1) Mehrdimensionale Klima-Bewertung (Hitze, Trockenheit, Winter, Variabilität)
2) Optionalen Standort-Faktor (Strasse/Park sowie zusätzliche GIS-/Katasterspalten)
3) Optionalen Management-Faktor (z. B. Bewässerung, Pflegeintensität)
4) Transparentere Ergebnisfelder für Bericht, Kontrolle und Plausibilisierung
5) Zusätzliche Ausgabedateien (Summary + Metadaten)

Grundidee:
- Pro Baum wird eine baseline Lebenserwartung pro Art angesetzt.
- Diese wird mit einem Urban-Faktor aus CityTrees-II angepasst.
- Danach wird ein Klima-Faktor berechnet:
    a) Proxy-Modell mit CitiesGOER + artspezifischer Vulnerabilität
    b) optional alternativ TreeGOER-Exceedance
- Optional werden Standort- und Management-Faktoren berücksichtigt.
- Daraus werden Restlebensdauer, erwartetes Ausfalljahr und Start Nachpflanzung abgeleitet.

Finale Logik:
    life_final = avg_life_baseline * urban_factor * climate_factor * site_factor * management_factor
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Hilfsfunktionen
# =========================

def normalize_species_name(s: str) -> str:
    """Normalisiert Artnamen auf Genus + species, entfernt Kultivare."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.replace("×", "x").replace(" X ", " x ").replace(" x ", " x ")
    s = re.sub(r"\s+'[^']+'", "", s)
    s = re.sub(r'\s+"[^"]+"', "", s)
    tokens = re.split(r"\s+", s)
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}".strip()
    return s


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def to_5y_window(year: Optional[float], base_year: int) -> str:
    if year is None or (isinstance(year, float) and math.isnan(year)) or pd.isna(year):
        return "unbekannt"
    y = int(year)
    block_start = base_year + ((y - base_year) // 5) * 5
    block_end = block_start + 4
    return f"{block_start}-{block_end}"


def norm_0_1(value: Optional[float], base: float, span: float, mode: str = "high_is_bad") -> float:
    """
    Normiert eine Größe in einen 0..1 Stressbereich.
    mode='high_is_bad'  -> Werte über base erzeugen Stress.
    mode='low_is_bad'   -> Werte unter base erzeugen Stress.
    """
    if value is None or span <= 0:
        return 0.0
    if mode == "high_is_bad":
        return max(0.0, min(1.0, (value - base) / span))
    if mode == "low_is_bad":
        return max(0.0, min(1.0, (base - value) / span))
    raise ValueError(f"Unbekannter mode für norm_0_1: {mode}")


def parse_factor_mapping(text: Optional[str]) -> Dict[str, float]:
    """
    Erwartet Format: 'strasse=0.90,park=1.00,platz=0.95'
    """
    mapping: Dict[str, float] = {}
    if not text:
        return mapping
    for part in str(text).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        try:
            mapping[key] = float(value.strip())
        except ValueError:
            continue
    return mapping


def map_text_factor(series: pd.Series, mapping: Dict[str, float], default: float = 1.0) -> pd.Series:
    if series is None or len(mapping) == 0:
        return pd.Series(default, index=range(len(series)) if series is not None else [])

    def _pick(v):
        if pd.isna(v):
            return default
        txt = str(v).strip().lower()
        for key, factor in mapping.items():
            if key in txt:
                return factor
        return default

    return series.apply(_pick)


# =========================
# Default-Lebensdauern Durchnitt aus Daten von Tom
# =========================

DEFAULT_BASELINE_LIFE: Dict[str, float] = {
    "Acer campestre": 125,
    "Acer platanoides": 125,
    "Acer pseudoplatanus": 125,
    "Aesculus hippocastanum": 105,
    "Alnus glutinosa": 60,
    "Amelanchier lamarckii": 60,
    "Betula pendula": 60,
    "Carpinus betulus": 150,
    "Castanea sativa": 135,
    "Corylus colurna": 150,
    "Fagus sylvatica": 155,
    "Fraxinus excelsior": 120,
    "Ginkgo biloba": 250,
    "Gleditsia triacanthos": 150,
    "Juglans regia": 160,
    "Liquidambar styraciflua": 150,
    "Pinus nigra": 150,
    "Platanus x acerifolia": 220,
    "Populus tremula": 85,
    "Prunus avium": 80,
    "Pyrus communis": 80,
    "Pyrus calleryana": 70,
    "Quercus robur": 180,
    "Quercus rubra": 180,
    "Robinia pseudoacacia": 120,
    "Salix alba": 85,
    "Sophora japonica": 150,
    "Sorbus intermedia": 120,
    "Tilia cordata": 180,
    "Tilia platyphyllos": 180,
    "Tilia europaea": 180,
    "Ulmus glabra": 130,
    "Ulmus hollandica": 130,
}


# =========================
# Reader
# =========================

def load_baseline_life(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return DEFAULT_BASELINE_LIFE.copy()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline-Life Datei nicht gefunden: {p}")
    t = pd.read_csv(p)
    if "species_norm" not in t.columns or "avg_life_baseline" not in t.columns:
        raise ValueError("Baseline-Life CSV braucht Spalten: species_norm, avg_life_baseline")
    d = DEFAULT_BASELINE_LIFE.copy()
    for _, row in t.iterrows():
        sp = normalize_species_name(row["species_norm"])
        val = safe_float(row["avg_life_baseline"])
        if sp and val is not None:
            d[sp] = val
    return d


def read_citiesgoer_excel(path: str, header_row: int = 6) -> pd.DataFrame:
    return pd.read_excel(path, header=header_row)


def pick_city_row(cities: pd.DataFrame, city_name: str, country_code: str) -> pd.Series:
    m = (cities["Name"].astype(str).str.lower() == city_name.lower()) & (cities["Country Code"] == country_code)
    sub = cities[m]
    if len(sub) == 0:
        sub = cities[cities["Name"].astype(str).str.contains(city_name, case=False, na=False)]
    if len(sub) == 0:
        raise ValueError(f"Stadt nicht gefunden in CitiesGOER: {city_name} ({country_code})")
    if "Population" in sub.columns:
        sub = sub.sort_values("Population", ascending=False)
    return sub.iloc[0]


def read_treegoer_csv(path: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    if "species" not in t.columns:
        raise ValueError("TreeGOER CSV braucht eine Spalte 'species'")
    t["species_norm"] = t["species"].apply(normalize_species_name)
    return t


# =========================
# Logik
# =========================

def classify_priority(row: pd.Series, current_year: int) -> Tuple[str, str, str]:
    if pd.isna(row.get("PFLANZJAHR_num")):
        return "niedrig", "Baumalter bestimmen", "Pflanzjahr fehlt"

    if pd.isna(row.get("life_final")):
        return "mittel", "Lebensdauer (Art) ergänzen", "Artwert fehlt"

    rest = row["life_final"] - (current_year - row["PFLANZJAHR_num"])
    climate_factor = row.get("climate_factor", 1.0)

    if rest <= 0:
        return "hoch", "Nachpflanzung starten", "Lebensdauer rechnerisch überschritten"
    if rest <= 20:
        return "hoch", "Nachpflanzung starten", "geringe Restlebensdauer"
    if climate_factor < 0.75:
        return "hoch", "Artenwahl und Nachpflanzung früh prüfen", "erhöhtes Klimarisiko"
    if rest <= 50:
        return "mittel", "Nachpflanzung einplanen", "mittlere Restlebensdauer"
    if climate_factor < 0.88:
        return "mittel", "Standort und Baumart beobachten", "moderates Klimarisiko"
    return "niedrig", "Beobachten / später prüfen", "aktuell kein dringender Handlungsbedarf"


# =========================
# Main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nachpflanzungsplanung Winterthur (Kataster + CityTrees II + Klima + Standort + optional TreeGOER)"
    )

    # Pflicht / allgemein
    ap.add_argument("--kataster", required=True, help="Pfad zum Baumkataster CSV.")
    ap.add_argument("--citytrees_scores", required=True, help="Pfad zu citytrees2_scores.csv (0..1 Scores).")
    ap.add_argument("--out_dir", default="out", help="Output-Verzeichnis.")
    ap.add_argument("--current_year", type=int, default=2026, help="Referenzjahr für Alter.")
    ap.add_argument("--lead_years", type=int, default=30, help="Vorlaufjahre für Nachpflanzung.")
    ap.add_argument("--baseline_life_csv", default=None, help="Optional: CSV (species_norm, avg_life_baseline).")

    # Katasterspalten
    ap.add_argument("--col_id", default="BAUMNUMMER", help="Baum-ID Spalte im Kataster.")
    ap.add_argument("--col_species", default="BAUMART_L", help="Arten-Spalte (lateinisch) im Kataster.")
    ap.add_argument("--col_plant_year", default="PFLANZJAHR", help="Pflanzjahr-Spalte im Kataster.")
    ap.add_argument("--col_category", default="KATEGORIE", help="Kategorie/Context (Park/Strasse) im Kataster.")
    ap.add_argument("--col_site_type", default=None, help="Optional: detaillierter Standorttyp.")
    ap.add_argument("--col_root_volume", default=None, help="Optional: Spalte mit Wurzelraum / Bodenvolumen.")
    ap.add_argument("--col_sealed", default=None, help="Optional: Spalte mit Versiegelung ja/nein.")
    ap.add_argument("--col_irrigation", default=None, help="Optional: Spalte mit Bewässerung ja/nein.")
    ap.add_argument("--col_maintenance", default=None, help="Optional: Spalte mit Pflegeintensität.")

    # CitiesGOER / Klima
    ap.add_argument("--citiesgoer_xlsx", default=None, help="Pfad zu CitiesGOER Excel.")
    ap.add_argument("--cities_header_row", type=int, default=6, help="Header-Zeile für CitiesGOER (0-index).")
    ap.add_argument("--city_name", default="Winterthur", help="Stadtname für CitiesGOER.")
    ap.add_argument("--country_code", default="CH", help="Country Code für CitiesGOER.")
    ap.add_argument("--scenario_label", default="2050s / SSP126", help="Freie Szenariobezeichnung für Bericht und Metadaten.")

    # Proxy-Klima aktivieren
    ap.add_argument("--use_proxy_climate", action="store_true", help="Aktiviere Proxy-Klima-Faktor (ohne TreeGOER).")

    # Klimaschwellen
    ap.add_argument("--proxy_heat_base", type=float, default=26.0, help="Schwelle ohne Hitzestress (bio05).")
    ap.add_argument("--proxy_heat_span", type=float, default=6.0, help="Spannweite bis Hitzestress=1.")
    ap.add_argument("--proxy_precip_base", type=float, default=900.0, help="Schwelle ohne Trockenstress (bio12).")
    ap.add_argument("--proxy_precip_span", type=float, default=600.0, help="Spannweite bis Trockenstress=1.")
    ap.add_argument("--proxy_winter_base", type=float, default=-8.0, help="Schwelle ohne Winterstress (bio06).")
    ap.add_argument("--proxy_winter_span", type=float, default=8.0, help="Spannweite bis Winterstress=1.")
    ap.add_argument("--proxy_variability_base", type=float, default=60.0, help="Schwelle ohne Variabilitätsstress (bio15).")
    ap.add_argument("--proxy_variability_span", type=float, default=40.0, help="Spannweite bis Variabilitätsstress=1.")
    ap.add_argument("--proxy_scale", type=float, default=2.5, help="Skalierung in exp(-scale * city_stress * species_vuln).")

    # Gewichte der Klima-Komponenten
    ap.add_argument("--weight_heat", type=float, default=0.40, help="Gewicht Hitze.")
    ap.add_argument("--weight_drought", type=float, default=0.35, help="Gewicht Trockenheit.")
    ap.add_argument("--weight_winter", type=float, default=0.15, help="Gewicht Winterstress.")
    ap.add_argument("--weight_variability", type=float, default=0.10, help="Gewicht Niederschlags-Variabilität.")

    # CityTrees-Vulnerabilität
    ap.add_argument("--species_weight_drought", type=float, default=0.35, help="Artgewicht Dürretoleranz.")
    ap.add_argument("--species_weight_water_need", type=float, default=0.20, help="Artgewicht Wasserbedarf.")
    ap.add_argument("--species_weight_heat", type=float, default=0.25, help="Artgewicht Hitzetoleranz.")
    ap.add_argument("--species_weight_frost", type=float, default=0.20, help="Artgewicht Frosttoleranz.")

    # TreeGOER
    ap.add_argument("--treegoer_csv", default=None, help="Optional: TreeGOER Species Ranges CSV.")
    ap.add_argument("--tree_var", default="bio05", help="BIO Variable für TreeGOER Exceedance.")
    ap.add_argument("--tree_q", default="q95", help="Quantil-Spalte suffix in TreeGOER.")
    ap.add_argument("--tree_k", type=float, default=0.08, help="k in exp(-k * exceedance).")
    ap.add_argument("--future_var_from_cities", default="bio05", help="Variable in CitiesGOER, die zu tree_var passt.")

    # Standort-Faktoren
    ap.add_argument(
        "--category_factor_map",
        default="strasse=0.90,straße=0.90,park=1.00",
        help="Textmapping für Kategorie-Faktor, z. B. 'strasse=0.90,park=1.00'.",
    )
    ap.add_argument(
        "--site_type_factor_map",
        default=None,
        help="Optionales Mapping für detaillierten Standorttyp, z. B. 'platz=0.92,allee=0.95,gruenanlage=1.00'.",
    )
    ap.add_argument("--sealed_yes_factor", type=float, default=0.92, help="Faktor falls Standort versiegelt.")
    ap.add_argument("--sealed_no_factor", type=float, default=1.00, help="Faktor falls Standort nicht versiegelt.")
    ap.add_argument("--root_volume_low_threshold", type=float, default=12.0, help="Schwelle kleiner Wurzelraum.")
    ap.add_argument("--root_volume_low_factor", type=float, default=0.90, help="Faktor bei kleinem Wurzelraum.")
    ap.add_argument("--root_volume_ok_factor", type=float, default=1.00, help="Faktor bei ausreichendem Wurzelraum.")

    # Management-Faktoren
    ap.add_argument("--irrigation_yes_factor", type=float, default=1.05, help="Faktor falls Bewässerung vorhanden.")
    ap.add_argument("--irrigation_no_factor", type=float, default=1.00, help="Faktor falls keine Bewässerung vorhanden.")
    ap.add_argument(
        "--maintenance_factor_map",
        default="hoch=1.05,mittel=1.00,niedrig=0.95",
        help="Optionales Mapping für Pflegeintensität.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Einlesen
    # =========================
    kat = pd.read_csv(args.kataster, sep=None, engine="python")
    scores = pd.read_csv(args.citytrees_scores)

    kat["species_norm"] = kat[args.col_species].apply(normalize_species_name)
    scores["species_norm"] = scores["species_latin"].apply(normalize_species_name)

    join_cols = [c for c in scores.columns if c != "species_latin"]
    df = kat.merge(scores[join_cols], on="species_norm", how="left")

    baseline = load_baseline_life(args.baseline_life_csv)
    df["avg_life_baseline"] = df["species_norm"].map(baseline)

    # =========================
    # Urban-Faktor aus CityTrees
    # =========================
    df["urban_overall"] = pd.to_numeric(df.get("urban_overall"), errors="coerce")
    df["urban_factor"] = np.where(df["urban_overall"].notna(), 0.8 + 0.4 * df["urban_overall"], 1.0)
    df["avg_life_after_urban"] = df["avg_life_baseline"] * df["urban_factor"]

    # =========================
    # Alter
    # =========================
    df["PFLANZJAHR_num"] = pd.to_numeric(df.get(args.col_plant_year), errors="coerce")
    df["age_current"] = (args.current_year - df["PFLANZJAHR_num"]).where(df["PFLANZJAHR_num"].notna())

    # =========================
    # Standort-Faktor
    # =========================
    category_map = parse_factor_mapping(args.category_factor_map)
    site_type_map = parse_factor_mapping(args.site_type_factor_map)
    maintenance_map = parse_factor_mapping(args.maintenance_factor_map)

    df["site_factor_category"] = 1.0
    if args.col_category in df.columns:
        df["site_factor_category"] = map_text_factor(df[args.col_category], category_map, default=1.0)

    df["site_factor_site_type"] = 1.0
    if args.col_site_type and args.col_site_type in df.columns and site_type_map:
        df["site_factor_site_type"] = map_text_factor(df[args.col_site_type], site_type_map, default=1.0)

    df["site_factor_sealing"] = 1.0
    if args.col_sealed and args.col_sealed in df.columns:
        sealed_txt = df[args.col_sealed].astype(str).str.lower()
        df["site_factor_sealing"] = np.where(
            sealed_txt.str.contains("ja|yes|true|1", regex=True, na=False),
            args.sealed_yes_factor,
            np.where(
                sealed_txt.str.contains("nein|no|false|0", regex=True, na=False),
                args.sealed_no_factor,
                1.0,
            ),
        )

    df["site_factor_root_volume"] = 1.0
    if args.col_root_volume and args.col_root_volume in df.columns:
        root_vol = pd.to_numeric(df[args.col_root_volume], errors="coerce")
        df["site_factor_root_volume"] = np.where(
            root_vol.notna() & (root_vol < args.root_volume_low_threshold),
            args.root_volume_low_factor,
            np.where(root_vol.notna(), args.root_volume_ok_factor, 1.0),
        )

    df["site_factor"] = (
        df["site_factor_category"]
        * df["site_factor_site_type"]
        * df["site_factor_sealing"]
        * df["site_factor_root_volume"]
    )

    # =========================
    # Management-Faktor
    # =========================
    df["management_factor_irrigation"] = 1.0
    if args.col_irrigation and args.col_irrigation in df.columns:
        irr_txt = df[args.col_irrigation].astype(str).str.lower()
        df["management_factor_irrigation"] = np.where(
            irr_txt.str.contains("ja|yes|true|1", regex=True, na=False),
            args.irrigation_yes_factor,
            np.where(
                irr_txt.str.contains("nein|no|false|0", regex=True, na=False),
                args.irrigation_no_factor,
                1.0,
            ),
        )

    df["management_factor_maintenance"] = 1.0
    if args.col_maintenance and args.col_maintenance in df.columns and maintenance_map:
        df["management_factor_maintenance"] = map_text_factor(df[args.col_maintenance], maintenance_map, default=1.0)

    df["management_factor"] = df["management_factor_irrigation"] * df["management_factor_maintenance"]

    # =========================
    # Klima-Faktor
    # =========================
    df["climate_factor"] = 1.0
    df["species_vulnerability_proxy"] = np.nan
    climate_note = "none"
    city_row = None

    if args.citiesgoer_xlsx:
        cities = read_citiesgoer_excel(args.citiesgoer_xlsx, header_row=args.cities_header_row)
        city_row = pick_city_row(cities, args.city_name, args.country_code)

    if args.use_proxy_climate:
        if city_row is None:
            raise ValueError("Proxy-Klima aktiv, aber --citiesgoer_xlsx fehlt.")

        bio05 = safe_float(city_row.get("bio05"))
        bio12 = safe_float(city_row.get("bio12"))
        bio06 = safe_float(city_row.get("bio06"))
        bio15 = safe_float(city_row.get("bio15"))

        if bio05 is None or bio12 is None:
            raise ValueError("CitiesGOER Stadtzeile enthält bio05/bio12 nicht oder nicht lesbar.")

        heat_stress = norm_0_1(bio05, args.proxy_heat_base, args.proxy_heat_span, mode="high_is_bad")
        drought_stress = norm_0_1(bio12, args.proxy_precip_base, args.proxy_precip_span, mode="low_is_bad")
        winter_stress = norm_0_1(bio06, args.proxy_winter_base, args.proxy_winter_span, mode="low_is_bad")
        variability_stress = norm_0_1(bio15, args.proxy_variability_base, args.proxy_variability_span, mode="high_is_bad")

        weight_sum = args.weight_heat + args.weight_drought + args.weight_winter + args.weight_variability
        if weight_sum <= 0:
            raise ValueError("Die Summe der Klima-Gewichte muss > 0 sein.")

        city_stress = (
            args.weight_heat * heat_stress
            + args.weight_drought * drought_stress
            + args.weight_winter * winter_stress
            + args.weight_variability * variability_stress
        ) / weight_sum

        # Artenvulnerabilität aus CityTrees-Feldern
        df["drought_tol"] = pd.to_numeric(df.get("drought_tol"), errors="coerce")
        df["low_water_need"] = pd.to_numeric(df.get("low_water_need"), errors="coerce")
        df["heat_tol"] = pd.to_numeric(df.get("heat_tol"), errors="coerce")
        df["frost_tol"] = pd.to_numeric(df.get("frost_tol"), errors="coerce")

        species_weight_sum = (
            args.species_weight_drought
            + args.species_weight_water_need
            + args.species_weight_heat
            + args.species_weight_frost
        )
        if species_weight_sum <= 0:
            raise ValueError("Die Summe der Arten-Gewichte muss > 0 sein.")

        species_vulnerability = (
            args.species_weight_drought * (1 - df["drought_tol"].fillna(0.5))
            + args.species_weight_water_need * (1 - df["low_water_need"].fillna(0.5))
            + args.species_weight_heat * (1 - df["heat_tol"].fillna(0.5))
            + args.species_weight_frost * (1 - df["frost_tol"].fillna(0.5))
        ) / species_weight_sum

        df["species_vulnerability_proxy"] = species_vulnerability.clip(lower=0.0, upper=1.0)
        df["city_heat_stress"] = heat_stress
        df["city_drought_stress"] = drought_stress
        df["city_winter_stress"] = winter_stress
        df["city_variability_stress"] = variability_stress
        df["city_stress"] = city_stress

        df["climate_factor"] = np.exp(-args.proxy_scale * city_stress * df["species_vulnerability_proxy"])
        climate_note = f"proxy_multifactor (scenario={args.scenario_label}; scale={args.proxy_scale})"

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

        df = df.merge(tree[["species_norm", q_col]], on="species_norm", how="left", suffixes=("", "_treegoer"))
        df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
        df["treegoer_exceedance"] = np.maximum(0.0, future_val - df[q_col])
        df["climate_factor"] = np.exp(-args.tree_k * df["treegoer_exceedance"])
        df.loc[df[q_col].isna(), "climate_factor"] = 1.0
        climate_note = f"treegoer_exceedance (scenario={args.scenario_label}; {q_col}; k={args.tree_k})"

    # =========================
    # Finale Lebenserwartung
    # =========================
    df["life_final"] = (
        df["avg_life_baseline"]
        * df["urban_factor"]
        * df["climate_factor"]
        * df["site_factor"]
        * df["management_factor"]
    )

    df["rest_life_years"] = np.where(
        df["life_final"].notna() & df["PFLANZJAHR_num"].notna(),
        df["life_final"] - (args.current_year - df["PFLANZJAHR_num"]),
        np.nan,
    )

    df["expected_failure_year"] = np.where(
        df["PFLANZJAHR_num"].notna() & df["life_final"].notna(),
        (df["PFLANZJAHR_num"] + df["life_final"]).round().astype("Int64"),
        pd.NA,
    )
    df["start_replant_year"] = np.where(
        df["expected_failure_year"].notna(),
        df["expected_failure_year"].astype("Int64") - args.lead_years,
        pd.NA,
    )
    df["massnahmenzeitraum"] = df["start_replant_year"].apply(lambda y: to_5y_window(y, args.current_year))

    prio, action, reason = zip(*df.apply(lambda row: classify_priority(row, args.current_year), axis=1))
    df["prioritaet"] = prio
    df["massnahme"] = action
    df["begruendung_modell"] = reason

    # =========================
    # Planungstabelle
    # =========================
    planning = pd.DataFrame({
        "Massnahmenzeitraum": df["massnahmenzeitraum"],
        "Gattung / Art": df[args.col_species],
        "Art (norm)": df["species_norm"],
        "Baum ID": df[args.col_id],
        "Kategorie": df.get(args.col_category, pd.Series(index=df.index, dtype="object")),
        "Pflanzjahr": df.get(args.col_plant_year).fillna("unbekannt"),
        f"Alter (Stand {args.current_year})": df["age_current"].round().astype("Int64").astype("object").where(df["age_current"].notna(), "unbekannt"),
        "Maßnahme": df["massnahme"],
        "Priorität": df["prioritaet"],
        "Modell-Begründung": df["begruendung_modell"],
        "Lebenserwartung Baseline": df["avg_life_baseline"].round(1).astype("object").where(df["avg_life_baseline"].notna(), "unbekannt"),
        "Urban-Faktor": df["urban_factor"].round(3),
        "Klima-Faktor": df["climate_factor"].round(3),
        "Standort-Faktor": df["site_factor"].round(3),
        "Management-Faktor": df["management_factor"].round(3),
        "Lebenserwartung final": df["life_final"].round(1).astype("object").where(df["life_final"].notna(), "unbekannt"),
        "Restlebensdauer": df["rest_life_years"].round(1).astype("object").where(df["rest_life_years"].notna(), "unbekannt"),
        "Start Nachpflanzung (Jahr)": df["start_replant_year"].astype("object").where(df["start_replant_year"].notna(), "unbekannt"),
        "Erwartetes Ausfalljahr": df["expected_failure_year"].astype("object").where(df["expected_failure_year"].notna(), "unbekannt"),
    })

    planning["Notizen"] = ""
    planning.loc[(df["expected_failure_year"].notna()) & (df["expected_failure_year"] <= args.current_year), "Notizen"] = (
        "Alter rechnerisch überschritten; fachliche Kontrolle empfohlen"
    )

    prio_order = {"hoch": 0, "mittel": 1, "niedrig": 2}
    planning["_prio_sort"] = planning["Priorität"].map(prio_order).fillna(9)
    planning["_start_sort"] = pd.to_numeric(planning["Start Nachpflanzung (Jahr)"], errors="coerce")
    planning = planning.sort_values(by=["_prio_sort", "_start_sort"], ascending=[True, True]).drop(columns=["_prio_sort", "_start_sort"])

    # =========================
    # Summary-Auswertung
    # =========================
    summary_rows = []
    summary_rows.append(["Anzahl Bäume", len(df)])
    summary_rows.append(["Gematchte CityTrees-Werte", int(df["urban_overall"].notna().sum())])
    summary_rows.append(["Gematchte Baseline-Lebensdauern", int(df["avg_life_baseline"].notna().sum())])
    summary_rows.append(["Hohe Priorität", int((df["prioritaet"] == "hoch").sum())])
    summary_rows.append(["Mittlere Priorität", int((df["prioritaet"] == "mittel").sum())])
    summary_rows.append(["Niedrige Priorität", int((df["prioritaet"] == "niedrig").sum())])
    summary_rows.append(["Mittlerer Klima-Faktor", round(float(df["climate_factor"].mean()), 3)])
    summary_rows.append(["Mittlerer Standort-Faktor", round(float(df["site_factor"].mean()), 3)])
    summary_rows.append(["Mittlerer Management-Faktor", round(float(df["management_factor"].mean()), 3)])
    summary_rows.append(["Mittlere finale Lebenserwartung", round(float(df["life_final"].dropna().mean()), 1) if df["life_final"].notna().any() else np.nan])
    summary_rows.append(["Szenario", args.scenario_label])
    summary_rows.append(["Klima-Modus", climate_note])
    summary = pd.DataFrame(summary_rows, columns=["Kennzahl", "Wert"])

    # =========================
    # Speichern
    # =========================
    out_csv = out_dir / "baumkataster_mit_scores_und_planung_v3.csv"
    out_xlsx = out_dir / "nachpflanzungsplanung_v3.xlsx"
    out_species_template = out_dir / "species_life_template_v3.csv"
    out_summary_csv = out_dir / "run_summary_v3.csv"
    out_meta = out_dir / "run_metadata_v3.txt"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        planning.to_excel(writer, index=False, sheet_name="Planung")
        summary.to_excel(writer, index=False, sheet_name="Summary")

    species_template = pd.DataFrame({"species_norm": sorted(df["species_norm"].dropna().unique())})
    species_template["avg_life_baseline"] = species_template["species_norm"].map(baseline)
    species_template["needs_review"] = species_template["avg_life_baseline"].isna()
    species_template.to_csv(out_species_template, index=False, encoding="utf-8")
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8")

    with open(out_meta, "w", encoding="utf-8") as f:
        f.write(f"scenario_label: {args.scenario_label}\n")
        f.write(f"climate_mode: {climate_note}\n")
        if city_row is not None:
            f.write(f"city: {args.city_name} {args.country_code}\n")
            for k in ["bio01", "bio05", "bio06", "bio12", "bio15"]:
                if k in city_row.index:
                    f.write(f"{k}: {city_row[k]}\n")
        f.write(f"rows: {len(df)}\n")
        f.write(f"matched_citytrees: {int(df['urban_overall'].notna().sum())}\n")
        f.write(f"matched_baseline_life: {int(df['avg_life_baseline'].notna().sum())}\n")
        f.write(f"high_priority: {int((df['prioritaet'] == 'hoch').sum())}\n")
        f.write(f"medium_priority: {int((df['prioritaet'] == 'mittel').sum())}\n")
        f.write(f"low_priority: {int((df['prioritaet'] == 'niedrig').sum())}\n")

    print("✅ Fertig.")
    print(f"- CSV:      {out_csv}")
    print(f"- XLSX:     {out_xlsx}")
    print(f"- Summary:  {out_summary_csv}")
    print(f"- Template: {out_species_template}")
    print(f"- Meta:     {out_meta}")


if __name__ == "__main__":
    import sys

    # Komfortmodus für Spyder / lokales Testen ohne Argumente
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--kataster", "Baumkataster_export_05.01.2026_mitStrassen_Parkbäumen.csv",
            "--citytrees_scores", "citytrees2_scores.csv",
            "--citiesgoer_xlsx", "CitiesGOER_2050s_ssp126.xlsx",
            "--use_proxy_climate",
            "--scenario_label", "2050s / SSP126",
            "--out_dir", "out",
        ])

    main()
