# Winterthur Stadtbaum-Modell v7 – Detaillierte Anleitung

## 1. Ziel des Modells

Dieses Modell simuliert die zukünftige Entwicklung des Winterthurer Stadtbaumbestands über 25, 50 und 100 Jahre. Es verbindet:

- historische Ausfalldaten aus dem Baumkataster,
- artspezifische Ausfallwahrscheinlichkeiten,
- CityTrees-II-Bewertungen,
- CitiesGOER-Klimaszenarien,
- TreeGOER-Klimahüllen,
- Standort- und Managementfaktoren,
- Ersatzpflanzungen,
- zusätzliche Neupflanzungen,
- Zielbestände, z. B. 17’000 Bäume bis 2030,
- Monte-Carlo-Unsicherheit,
- optional artspezifische Lebensdauer-Bandbreiten.

Das Modell beantwortet vor allem Fragen wie:

- Wie entwickelt sich der Baumbestand in 25, 50 und 100 Jahren?
- Welche Nachpflanzungsrate ist nötig, um 17’000 Bäume bis 2030 zu erreichen?
- Wie stark beeinflussen Klimaszenarien, TreeGOER, Ersatzrate und Pflanzstrategie die Ergebnisse?
- Welche Unsicherheit entsteht durch zufällige Ausfälle und durch unsichere Lebensdauerannahmen?

---

## 2. Grundidee des stochastischen Modells

Das Modell ist ein Monte-Carlo-Modell. Es simuliert nicht einen einzigen festen Zukunftspfad, sondern viele mögliche Zukunftspfade.

Für jedes Simulationsjahr und jeden Baum wird eine jährliche Ausfallwahrscheinlichkeit berechnet. Diese basiert auf:

```text
kalibrierte Hazard aus historischen Fällungen
× Klimaeffekt
× Standorteffekt
× Managementeffekt
× Urbanitäts-/CityTrees-Effekt
× Kategorieeffekt
× optionaler Lebensdauer-Multiplikator
```

Dann wird pro Baum per Zufallszahl entschieden, ob er in diesem Jahr ausfällt.

Durch viele Wiederholungen (`n_runs`) entstehen Mittelwerte und Unsicherheitsbereiche:

- `mean`: Mittelwert über alle Läufe
- `p05`: 5%-Perzentil
- `p25`: 25%-Perzentil
- `p50`: Median
- `p75`: 75%-Perzentil
- `p95`: 95%-Perzentil
- `min`, `max`: Minimum und Maximum über alle Läufe

---

## 3. Unterschied zum statischen Planungsmodell

Das frühere statische Modell arbeitet deterministisch mit einer Lebenserwartung:

```text
life_final = avg_life_baseline × urban_factor × climate_factor × site_factor × management_factor
```

Daraus wird ein erwartetes Ausfalljahr berechnet.

Das stochastische Modell arbeitet anders:

```text
jährliche Ausfallwahrscheinlichkeit = kalibrierte Hazard × Risikomultiplikatoren
```

Das bedeutet:

- Der statische Code sagt eher: „Wann ist ein Baum rechnerisch am Lebensende?“
- Der MC-Code sagt eher: „Mit welcher Wahrscheinlichkeit fällt ein Baum in jedem Jahr aus?“

Wenn man die Zufälligkeit ausschaltet, muss das MC-Modell daher nicht exakt dasselbe Resultat liefern wie das statische Modell. Es sollte aber im Erwartungswert plausibel und konsistent sein.

---

## 4. Ordnerstruktur

Empfohlene Projektstruktur:

```text
stadtbaeume_stochastic_project/
├─ scripts/
│  ├─ winterthur_tree_stochastic_goal_planning_v7.py
│  └─ run_winterthur_scenarios_v7.py
│
├─ configs/
│  ├─ base_target_17000_v7_life_ranges.yaml
│  └─ scenario_grid_v7_life_ranges.yaml
│
├─ data/
│  ├─ 2026-04-13_Baumkataster_gesamte_Daten.csv
│  ├─ citytrees2_scores.csv
│  ├─ CitiesGOER_2050s_ssp126.xlsx
│  ├─ TreeGOER_2023_wide.csv
│  └─ species_life_ranges.csv
│
└─ output/
   └─ runs/
```

---

## 5. Benötigte Eingabedateien

### 5.1 Baumkataster

Beispiel:

```text
data/2026-04-13_Baumkataster_gesamte_Daten.csv
```

Das Kataster sollte mindestens enthalten:

| Spalte | Bedeutung |
|---|---|
| `BAUMNUMMER` | eindeutige Baum-ID |
| `BAUMSTATUS` | Status, z. B. Aktiv / Gefällt |
| `BAUMART_L` | lateinischer Artname |
| `PFLANZJAHR` | Pflanzjahr |
| `FALLDATUM` | Fälldatum bei gefällten Bäumen |
| `BAUMTYP` oder `KATEGORIE` | Baumkategorie, z. B. Strassenbaum / Parkbaum |

Falls deine Spalten anders heißen, kannst du die Spaltennamen in der YAML-Datei anpassen.

---

### 5.2 CityTrees-II Scores

Beispiel:

```text
data/citytrees2_scores.csv
```

Diese Datei liefert artspezifische Eignungswerte wie:

- `urban_overall`
- `drought_tol`
- `low_water_need`
- `heat_tol`
- `frost_tol`

Wichtig ist eine Artenspalte, meistens:

```text
species_latin
```

---

### 5.3 CitiesGOER-Klimadaten

Beispiel:

```text
data/CitiesGOER_2050s_ssp126.xlsx
```

Diese Datei liefert zukünftige Klimawerte für Winterthur, z. B.:

- `bio05`: maximale Temperatur des wärmsten Monats
- `bio06`: minimale Temperatur des kältesten Monats
- `bio12`: Jahresniederschlag
- `bio15`: Niederschlagsvariabilität

Wichtige Parameter:

```yaml
citiesgoer_xlsx: data/CitiesGOER_2050s_ssp126.xlsx
cities_header_row: 6
city_name: Winterthur
country_code: CH
scenario_label: 2050s / SSP126
```

---

### 5.4 TreeGOER-Datei

Beispiel:

```text
data/TreeGOER_2023_wide.csv
```

Die Datei muss im Wide-Format vorliegen. Für die Default-Einstellung braucht das Modell z. B. die Spalte:

```text
bio05_q95
```

Falls du ursprünglich eine Long-Format-Datei hast mit Spalten wie:

```text
species, var, MIN, QRT1, MEDIAN, MEAN, QRT3, MAX, Q05, Q95
```

muss diese zuerst ins Wide-Format umgewandelt werden.

---

### 5.5 Lebensdauer-Bandbreiten pro Baumart

Beispiel:

```text
data/species_life_ranges.csv
```

Empfohlene Struktur:

```csv
species_norm,life_min,life_mode,life_max
Acer campestre,90,125,150
Betula pendula,40,60,80
Ginkgo biloba,180,250,300
Platanus x acerifolia,150,220,280
Tilia cordata,130,180,230
```

Bedeutung:

| Spalte | Bedeutung |
|---|---|
| `species_norm` | normalisierter lateinischer Artname |
| `life_min` | plausible untere Lebensdauer |
| `life_mode` | typischster / wahrscheinlichster Wert |
| `life_max` | plausible obere Lebensdauer |

`life_mode` ist der wahrscheinlichste Wert. Bei einer Dreiecksverteilung werden Werte nahe `life_mode` häufiger gezogen als Werte nahe `life_min` oder `life_max`.

---

## 6. YAML-Konfiguration

Statt sehr langer Command-Line-Befehle wird das Modell über YAML-Dateien gesteuert.

Beispiel:

```yaml
kataster: data/2026-04-13_Baumkataster_gesamte_Daten.csv
citytrees_scores: data/citytrees2_scores.csv
citiesgoer_xlsx: data/CitiesGOER_2050s_ssp126.xlsx
treegoer_csv: data/TreeGOER_2023_wide.csv
baseline_life_range_csv: data/species_life_ranges.csv

use_proxy_climate: true
scenario_label: 2050s / SSP126

out_dir: output/runs/base_target_17000

years: 100
n_runs: 300
milestones: [4, 25, 50, 100]

current_year: 2026
target_count: 17000

replacement_rate: 0.8
replacement_delay: 2

annual_new_trees: 300
annual_new_trees_start_offset: 1
annual_new_trees_end_offset: 4
new_tree_strategy: balanced

tree_var: bio05
tree_q: q95
tree_k: 0.08
future_var_from_cities: bio05

life_sampling: triangular
life_uncertainty_mode: per_run
life_hazard_adjustment_weight: 0.5
life_hazard_reference: 0.0
life_hazard_min_multiplier: 0.5
life_hazard_max_multiplier: 2.0
```

---

## 7. Wichtigste Parameter

### 7.1 Simulationsdauer

```yaml
years: 100
milestones: [4, 25, 50, 100]
```

Bedeutung bei `current_year: 2026`:

| `year_offset` | Kalenderjahr |
|---:|---:|
| 4 | 2030 |
| 25 | 2051 |
| 50 | 2076 |
| 100 | 2126 |

---

### 7.2 Anzahl Monte-Carlo-Läufe

```yaml
n_runs: 300
```

Empfehlung:

| Zweck | Wert |
|---|---:|
| schneller Funktionstest | 10–50 |
| Szenario-Vergleich | 100–300 |
| finale Resultate | 300–1000 |

Je höher `n_runs`, desto stabiler die Perzentile, aber desto länger dauert die Berechnung.

---

### 7.3 Zielbestand

```yaml
target_count: 17000
```

Das Modell wertet aus, ob und wann dieser Bestand erreicht wird.

Wichtige Output-Dateien:

```text
target_summary.csv
target_first_hit_by_run.csv
```

---

### 7.4 Ersatzpflanzungen

```yaml
replacement_rate: 0.8
replacement_delay: 2
```

Bedeutung:

| Parameter | Bedeutung |
|---|---|
| `replacement_rate` | Anteil ausgefallener Bäume, die ersetzt werden |
| `replacement_delay` | Jahre zwischen Ausfall und Ersatzpflanzung |

Beispiele:

```yaml
replacement_rate: 0.8
replacement_delay: 2
```

80 % der Ausfälle werden nach zwei Jahren ersetzt.

```yaml
replacement_rate: 1.0
replacement_delay: 1
```

Alle Ausfälle werden nach einem Jahr ersetzt.

---

### 7.5 Zusätzliche Neupflanzungen

```yaml
annual_new_trees: 300
annual_new_trees_start_offset: 1
annual_new_trees_end_offset: 4
```

Bedeutung bei `current_year: 2026`:

| Offset | Jahr |
|---:|---:|
| 1 | 2027 |
| 2 | 2028 |
| 3 | 2029 |
| 4 | 2030 |

Das Beispiel bedeutet:

> Von 2027 bis 2030 werden jedes Jahr zusätzlich 300 Bäume gepflanzt.

Das ist zentral für das Ziel:

```text
17’000 Bäume bis 2030
```

---

### 7.6 Pflanzstrategie für neue Bäume

```yaml
new_tree_strategy: balanced
```

Mögliche Werte:

| Strategie | Bedeutung |
|---|---|
| `same_mix` | neue Bäume folgen ungefähr der heutigen Artverteilung |
| `long_life` | bevorzugt langlebige Arten |
| `climate_fit` | bevorzugt Arten mit gutem Klima-/TreeGOER-Fit |
| `balanced` | kombiniert Langlebigkeit, Klimaeignung und Diversität |

Empfehlung:

- Für Vergleich mit Status quo: `same_mix`
- Für strategische Zukunftsplanung: `balanced`
- Für Sensitivitätstest: alle Strategien vergleichen

---

## 8. Lebensdauer-Bandbreiten und Hazard-Anpassung

### 8.1 Ziel

In v7 können artspezifische Lebensdauer-Bandbreiten direkt die jährliche Ausfallwahrscheinlichkeit beeinflussen.

Beispiel:

```csv
species_norm,life_min,life_mode,life_max
Betula pendula,40,60,80
Ginkgo biloba,180,250,300
```

Eine kurzlebige Art wie `Betula pendula` erhält tendenziell eine höhere Hazard. Eine langlebige Art wie `Ginkgo biloba` erhält tendenziell eine tiefere Hazard.

---

### 8.2 Sampling-Methode

```yaml
life_sampling: triangular
```

Das Modell zieht Lebensdauerwerte aus einer Dreiecksverteilung:

```text
life_min ---- life_mode ---- life_max
```

Werte nahe `life_mode` sind am wahrscheinlichsten.

---

### 8.3 Unsicherheitsmodus

```yaml
life_uncertainty_mode: per_run
```

Mögliche Werte:

| Modus | Bedeutung |
|---|---|
| `none` | keine zufällige Lebensdauerziehung; typischer Wert wird verwendet |
| `per_run` | pro Monte-Carlo-Lauf wird pro Art ein Lebensdauerwert gezogen |
| `per_tree` | jeder Baum bekommt einen eigenen gezogenen Lebensdauerwert |

Empfehlung:

```yaml
life_uncertainty_mode: per_run
```

Das bedeutet:

> In jedem Simulationslauf gilt eine plausible Lebensdauerannahme pro Art. Zwischen den Läufen variiert diese Annahme.

Das ist gut geeignet für Literatur- und Expertenschätzungen.

---

### 8.4 Stärke der Hazard-Anpassung

```yaml
life_hazard_adjustment_weight: 0.5
```

Dieser Parameter bestimmt, wie stark die Lebensdauerannahmen die empirisch kalibrierte Hazard verändern.

| Wert | Interpretation |
|---:|---|
| 0.0 | Effekt ausgeschaltet |
| 0.3 | schwacher Effekt |
| 0.5 | mittlerer Effekt |
| 1.0 | starker Effekt |

Empfehlung für erste Läufe:

```yaml
life_hazard_adjustment_weight: 0.3
```

oder:

```yaml
life_hazard_adjustment_weight: 0.5
```

Für Sensitivitätsanalyse:

```yaml
life_hazard_adjustment_weight: [0.0, 0.3, 0.5, 1.0]
```

---

### 8.5 Begrenzung des Multiplikators

```yaml
life_hazard_min_multiplier: 0.5
life_hazard_max_multiplier: 2.0
```

Damit wird verhindert, dass Lebensdauerannahmen die empirische Hazard zu extrem verändern.

Beispiel:

- `0.5`: Hazard kann höchstens halbiert werden
- `2.0`: Hazard kann höchstens verdoppelt werden

Empfehlung:

```yaml
life_hazard_min_multiplier: 0.5
life_hazard_max_multiplier: 2.0
```

---

## 9. Klima- und TreeGOER-Parameter

### 9.1 Proxy-Klima

```yaml
use_proxy_climate: true
```

Der Proxy-Klimafaktor kombiniert CitiesGOER und CityTrees:

- Hitzestress
- Trockenstress
- Winterstress
- Variabilitätsstress
- artspezifische Vulnerabilität

Wichtige Parameter:

```yaml
proxy_scale: 2.5
weight_heat: 0.40
weight_drought: 0.35
weight_winter: 0.15
weight_variability: 0.10
```

---

### 9.2 TreeGOER

```yaml
treegoer_csv: data/TreeGOER_2023_wide.csv
tree_var: bio05
tree_q: q95
tree_k: 0.08
future_var_from_cities: bio05
```

Das Modell vergleicht den zukünftigen CitiesGOER-Wert mit der artspezifischen TreeGOER-Grenze.

Beispiel:

```text
future bio05 von Winterthur - bio05_q95 der Art
```

Wenn der zukünftige Wert über der Artgrenze liegt, entsteht ein Exceedance-Stress.

---

### 9.3 TreeGOER-Sensitivität

Wichtige Varianten:

```yaml
tree_q: q95
```

weniger streng, da hohe Artgrenze.

```yaml
tree_q: qrt3
```

strenger, da niedrigere Artgrenze.

```yaml
tree_k: 0.08
```

moderater Effekt.

```yaml
tree_k: 0.15
```

stärkerer Effekt.

Empfohlene Sensitivität:

```yaml
tree_q: [q95, qrt3]
tree_k: [0.05, 0.08, 0.15]
```

---

## 10. Output-Dateien eines Einzellaufs

Typischer Output-Ordner:

```text
output/runs/20260430_143512_target17000_balanced
```

Wichtige Dateien:

| Datei | Inhalt |
|---|---|
| `calibrated_monte_carlo_results_v7.xlsx` | Excel-Sammeldatei mit wichtigsten Sheets |
| `total_summary.csv` | Bestandsentwicklung pro Jahr mit Perzentilen |
| `total_milestones.csv` | Bestände bei Meilensteinen, z. B. 2030, 2051, 2076, 2126 |
| `annual_summary.csv` | jährliche Ausfälle und Ersatzpflanzungen |
| `species_summary.csv` | artspezifische Entwicklung |
| `species_milestones.csv` | artspezifische Entwicklung an Meilensteinen |
| `target_summary.csv` | Erreichung des Zielbestands |
| `target_first_hit_by_run.csv` | erstes Zieljahr je Monte-Carlo-Lauf |
| `factor_diagnostics.csv` | Diagnose zu Klima-/Standort-/Managementfaktoren |
| `climate_components.csv` | Klimakomponenten und TreeGOER-Modus |
| `run_metadata.csv` | verwendete Parameter |

---

## 11. Interpretation der wichtigsten Outputs

### 11.1 `total_milestones.csv`

Beispielspalten:

```text
year_offset, year, mean, p05, p25, p50, p75, p95, min, max
```

Für Ziel 2030 ist wichtig:

```text
year_offset = 4
```

Bei `current_year = 2026` bedeutet das:

```text
2026 + 4 = 2030
```

Interpretation:

| Wert | Bedeutung |
|---|---|
| `mean >= 17000` | Ziel im Durchschnitt erreicht |
| `p50 >= 17000` | Ziel in mindestens 50 % der Läufe erreicht |
| `p05 >= 17000` | Ziel sehr robust erreicht; 95 % der Läufe liegen darüber |

Für politische Planung ist `p50 >= 17000` eine mittlere Zielerreichung. Für robuste Planung ist `p05 >= 17000` strenger.

---

### 11.2 `target_summary.csv`

Diese Datei beantwortet:

- Wird der Zielbestand erreicht?
- In wie vielen Läufen?
- In welchem Jahr im Mittel?
- Wie robust ist das Ziel?

---

### 11.3 `species_milestones.csv`

Diese Datei zeigt, welche Arten langfristig zunehmen oder abnehmen.

Wichtig für:

- Diversitätsanalyse
- Risikoarten
- klimaangepasste Nachpflanzungsstrategie
- Artenzusammensetzung in 25, 50 und 100 Jahren

---

## 12. Einzelszenario starten

In Spyder, wenn dein Working Directory der Projektordner ist:

```python
%runfile scripts/winterthur_tree_stochastic_goal_planning_v7.py --args "--config configs/base_target_17000_v7_life_ranges.yaml"
```

Oder im Terminal:

```bash
python scripts/winterthur_tree_stochastic_goal_planning_v7.py --config configs/base_target_17000_v7_life_ranges.yaml
```

---

## 13. Viele Szenarien mit Master-Runner starten

Der Master-Runner liest eine Grid-YAML und startet viele Kombinationen.

Spyder:

```python
%runfile scripts/run_winterthur_scenarios_v7.py --args "--scenario_config configs/scenario_grid_v7_life_ranges.yaml"
```

Terminal:

```bash
python scripts/run_winterthur_scenarios_v7.py --scenario_config configs/scenario_grid_v7_life_ranges.yaml
```

---

## 14. Beispiel für Szenario-Grid

```yaml
base_config: configs/base_target_17000_v7_life_ranges.yaml
output_root: output/scenario_runs
summary_csv: output/scenario_summary_v7.csv
sqlite_db: output/scenario_results_v7.sqlite

parameters:
  annual_new_trees: [250, 300, 350]
  replacement_rate: [0.8, 1.0]
  replacement_delay: [1, 2]
  new_tree_strategy: [same_mix, balanced]
  tree_q: [q95, qrt3]
  tree_k: [0.08, 0.15]
  climate_trend_end: [0.25, 0.5]
  life_hazard_adjustment_weight: [0.0, 0.3, 0.5]
  life_uncertainty_mode: [none, per_run]
```

Achtung: Die Anzahl Läufe wächst schnell.

Beispiel:

```text
3 × 2 × 2 × 2 × 2 × 2 × 2 × 3 × 2 = 1152 Szenarien
```

Wenn jedes Szenario `300` Monte-Carlo-Läufe macht, ist das sehr rechenintensiv.

Für erste Tests besser klein starten:

```yaml
parameters:
  annual_new_trees: [250, 300, 350]
  replacement_rate: [0.8, 1.0]
  new_tree_strategy: [same_mix, balanced]
  tree_q: [q95]
  tree_k: [0.08]
  climate_trend_end: [0.25]
  life_hazard_adjustment_weight: [0.0, 0.5]
  life_uncertainty_mode: [none, per_run]
```

Das ergibt:

```text
3 × 2 × 2 × 1 × 1 × 1 × 2 × 2 = 48 Szenarien
```

---

## 15. Empfohlener Arbeitsablauf

### Schritt 1: Funktionstest

```yaml
n_runs: 10
annual_new_trees: 300
```

Ziel: Prüfen, ob das Skript sauber durchläuft.

---

### Schritt 2: Erste Szenarien

```yaml
n_runs: 50
annual_new_trees: [250, 300, 350]
new_tree_strategy: [same_mix, balanced]
```

Ziel: grobe Orientierung.

---

### Schritt 3: Sensitivitätsanalyse

```yaml
n_runs: 100
replacement_rate: [0.8, 1.0]
tree_q: [q95, qrt3]
tree_k: [0.08, 0.15]
life_hazard_adjustment_weight: [0.0, 0.3, 0.5]
```

Ziel: wichtigste Treiber erkennen.

---

### Schritt 4: Finale Läufe

Für ausgewählte Szenarien:

```yaml
n_runs: 300
```

oder:

```yaml
n_runs: 500
```

Ziel: stabile finale Perzentile.

---

## 16. Empfehlungen für Ziel 17’000 Bäume bis 2030

Aus den bisherigen Läufen ergibt sich grob:

- Ohne zusätzliche Neupflanzungen sinkt der Bestand.
- Ersatzpflanzungen allein reichen nicht, um 17’000 Bäume zu erreichen.
- Zusätzliche Neupflanzungen von ca. 250–350 Bäumen/Jahr bis 2030 sind ein sinnvoller Testbereich.

Empfohlene Szenarien:

| Szenario | annual_new_trees | Strategie | Ersatzrate | Ziel |
|---|---:|---|---:|---|
| konservativ | 250 | balanced | 0.8 | wahrscheinlich knapp |
| mittel | 300 | balanced | 0.8 | wahrscheinlich erreichbar |
| robust | 350 | balanced | 0.8 | mit Puffer |
| maximal | 300 | balanced | 1.0 | Ersatz vollständig plus Zusatzpflanzung |

Für die Bewertung:

- `mean >= 17000`: durchschnittlich erreicht
- `p50 >= 17000`: in der Hälfte der Läufe erreicht
- `p05 >= 17000`: robust erreicht

---

## 17. Deterministischer Modus

Falls im v7-Skript aktiviert, kann ein deterministischer Modus genutzt werden:

```yaml
deterministic: true
```

Dieser Modus schaltet die binäre Zufallsziehung aus und arbeitet stärker mit Erwartungswerten.

Wichtig:

Der deterministische Modus ist nicht identisch mit dem alten statischen Planungsmodell. Er ist die deterministische Variante des Monte-Carlo-Hazard-Modells.

Er eignet sich für:

- Plausibilitätschecks,
- Vergleich mit MC-Mittelwerten,
- Kommunikation der Modelllogik.

---

## 18. Häufige Fehler und Lösungen

### 18.1 `FileNotFoundError`

Ursache: Pfad stimmt nicht relativ zum Working Directory.

Lösung:

- Prüfe in Spyder:

```python
import os
print(os.getcwd())
```

- Wenn Working Directory Projektordner ist, nutze:

```text
data/...
scripts/...
configs/...
```

- Wenn Working Directory `scripts` ist, nutze:

```text
../data/...
../configs/...
```

Empfehlung: Working Directory immer auf Projektordner setzen.

---

### 18.2 `UnicodeDecodeError`

Ursache: CSV-Datei ist nicht UTF-8 codiert.

Lösung: Datei nach UTF-8 konvertieren oder robuste Reader-Funktion nutzen.

---

### 18.3 `TreeGOER Spalte nicht gefunden: bio05_q95`

Ursache: TreeGOER-Datei ist nicht im Wide-Format oder Spalte fehlt.

Lösung: TreeGOER ins Wide-Format umwandeln und prüfen:

```python
import pandas as pd
x = pd.read_csv("data/TreeGOER_2023_wide.csv")
print("bio05_q95" in x.columns)
```

---

### 18.4 Sehr lange Laufzeit

Ursachen:

- zu viele Szenarien,
- zu viele MC-Läufe,
- 100 Jahre Simulation,
- viele Arten-/Jahresoutputs.

Lösung:

- zuerst mit `n_runs: 10` testen,
- dann `n_runs: 50`,
- Grid klein halten,
- finale Läufe nur für ausgewählte Szenarien.

---

## 19. Methodische Hinweise

### 19.1 Was das Modell gut kann

Das Modell eignet sich gut für:

- strategische Prognosen,
- Vergleich von Nachpflanzungsstrategien,
- Sensitivitätsanalysen,
- Unsicherheitskommunikation,
- Zielerreichung 2030,
- Langfristvergleich 25/50/100 Jahre.

---

### 19.2 Was das Modell nicht exakt leisten kann

Das Modell sagt nicht:

- welcher konkrete Einzelbaum exakt wann ausfällt,
- welche Pflanzung an welchem Standort im Detail optimal ist,
- wie sich politische oder operative Einschränkungen exakt auswirken.

Es ist ein strategisches Szenario- und Risikomodell.

---

### 19.3 Wichtige Annahmen transparent machen

Für Berichte immer dokumentieren:

- Startjahr und Startbestand,
- Klimaszenario,
- TreeGOER-Variable und Quantil,
- Ersatzrate,
- Ersatzverzögerung,
- zusätzliche Pflanzungen pro Jahr,
- Pflanzstrategie,
- Lebensdauer-Bandbreiten,
- Anzahl Monte-Carlo-Läufe,
- Zielkriterium (`mean`, `p50`, `p05`).

---

## 20. Beispiel-Formulierung für Bericht

> Das stochastische Modell simuliert die Entwicklung des Winterthurer Stadtbaumbestands ausgehend vom Katasterbestand 2026. Die jährlichen Ausfallwahrscheinlichkeiten werden aus historischen Fällungen kalibriert und mit artspezifischen Klima-, Standort-, Management- und Urbanitätsfaktoren modifiziert. Zusätzlich werden TreeGOER-Klimahüllen und artspezifische Lebensdauer-Bandbreiten berücksichtigt. Durch Monte-Carlo-Simulationen entstehen Erwartungswerte und Unsicherheitsintervalle für die Bestandsentwicklung in 25, 50 und 100 Jahren. Für das Ziel von 17’000 Bäumen bis 2030 werden unterschiedliche Nachpflanzungsraten und Pflanzstrategien getestet.

---

## 21. Empfohlene finale Ergebnisdarstellung

Für die Präsentation oder den Bericht sind folgende Tabellen/Grafiken sinnvoll:

1. Bestandsentwicklung Gesamtbestand bis 2126
2. Vergleich der Szenarien 2030 / 2051 / 2076 / 2126
3. Zielerreichung 17’000 Bäume bis 2030
4. Sensitivität gegenüber `annual_new_trees`
5. Sensitivität gegenüber `replacement_rate`
6. Sensitivität gegenüber `tree_q` und `tree_k`
7. Sensitivität gegenüber `life_hazard_adjustment_weight`
8. Artenzusammensetzung in 25 / 50 / 100 Jahren
9. Unsicherheitsband p05–p95 je Szenario

---

## 22. Minimaler Startpunkt

Für den ersten sauberen Test:

```yaml
n_runs: 50
years: 100
milestones: [4, 25, 50, 100]
target_count: 17000
annual_new_trees: 300
annual_new_trees_start_offset: 1
annual_new_trees_end_offset: 4
new_tree_strategy: balanced
replacement_rate: 0.8
replacement_delay: 2
tree_q: q95
tree_k: 0.08
life_uncertainty_mode: per_run
life_hazard_adjustment_weight: 0.5
```

Wenn dieser Lauf plausibel ist, danach mit `n_runs: 300` wiederholen.

---

## 23. Kurzfazit

Das v7-Modell ist dafür gedacht, systematisch zu prüfen:

```text
Welche Kombination aus Nachpflanzungsrate, Ersatzstrategie, Klimaannahme, TreeGOER-Schwellenwert und Lebensdauerannahmen führt dazu, dass Winterthur 2030 17’000 Bäume erreicht und der Bestand auch langfristig stabil bleibt?
```

Die zentrale Stärke des Modells ist nicht ein einzelnes Ergebnis, sondern der Vergleich vieler plausibler Szenarien mit Unsicherheitsbandbreiten.
