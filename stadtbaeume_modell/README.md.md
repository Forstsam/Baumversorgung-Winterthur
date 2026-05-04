# Stadtbaum-Modell – Winterthur

## Projektbeschreibung

Dieses Projekt dient zur **modellgestützten Bewertung und Planung von Stadtbäumen** basierend auf einem Baumkataster.

Ziel ist es, die **zukünftige Entwicklung des Baumbestands** abzuschätzen und daraus eine **Nachpflanzungsplanung** abzuleiten.

Dabei werden pro Baum folgende Aspekte berücksichtigt:

* Baumart und Alter
* Standortbedingungen (z. B. Straße / Park)
* Klimastress (zukünftige Szenarien)
* Pflege und Management

---

## Modelllogik

Die zentrale Idee des Modells ist:

> Die Lebensdauer eines Baumes wird durch verschiedene Einflussfaktoren angepasst.

Die finale Lebenserwartung berechnet sich als:

```
life_final = avg_life_baseline × urban_factor × climate_factor × site_factor × management_factor
```

---

## Klima-Faktor (climate_factor)

Der Klima-Faktor beschreibt den Einfluss des zukünftigen Klimas auf die Lebensdauer eines Baumes.

Er wird im Proxy-Modell wie folgt berechnet:

```
climate_factor = exp(-proxy_scale × city_stress × species_vulnerability)
```

### 🔹 Bestandteile:

* **city_stress**
  → Klimastress der Stadt (Hitze, Trockenheit, Winter, Variabilität)
  → basiert auf CitiesGOER-Daten

* **species_vulnerability**
  → Empfindlichkeit der Baumart
  → basiert auf CityTrees-Daten (z. B. Trockenheitstoleranz)

* **proxy_scale**
  → Skalierungsparameter (Standard: 2.5)

### Interpretation:

| Wert | Bedeutung         |
| ---- | ----------------- |
| 1.0  | kein Klimaeffekt  |
| 0.9  | leichte Reduktion |
| 0.7  | starke Reduktion  |

👉 Je höher Stress und Empfindlichkeit, desto stärker sinkt die Lebensdauer.

---

## Weitere Faktoren

### Urban-Faktor

* basiert auf CityTrees (Stadteignung)
* verbessert oder reduziert Lebensdauer

### Standort-Faktor

* berücksichtigt:

  * Straße vs. Park
  * Versiegelung
  * Wurzelraum

### Management-Faktor

* berücksichtigt:

  * Bewässerung
  * Pflegeintensität

---

## Ergebnisse

Das Modell berechnet für jeden Baum:

* finale Lebenserwartung
* Restlebensdauer
* erwartetes Ausfalljahr
* Startzeitpunkt für Nachpflanzung

Zusätzlich erfolgt eine **Priorisierung**:

* 🔴 hoch → sofortiger Handlungsbedarf
* 🟠 mittel → mittelfristig
* 🟢 niedrig → aktuell stabil

---

## Projektstruktur

```
stadtbaeume_modell/
│
├── data/        # Eingabedaten (Beispieldaten)
├── scripts/     # Python-Skripte
├── output/      # Ergebnisse
├── README.md
└── .gitignore
```

---

## Verwendung

### Voraussetzungen

* Python 3.x
* Bibliotheken:

  * pandas
  * numpy
  * openpyxl

---

### Beispielaufruf

```
python scripts/winterthur_tree_planning_v3_climate_extended.py \
--kataster data/example_baumkataster.csv \
--citytrees_scores data/citytrees2_scores.csv \
--citiesgoer_xlsx data/CitiesGOER_2050s_ssp126.xlsx \
--use_proxy_climate
```

---

## Hinweise

* Das Modell ist eine **vereinfachte Simulation**, keine exakte Prognose
* Ergebnisse sind als **strategische Entscheidungsgrundlage** gedacht
* Parameter können angepasst und kalibriert werden

---

## Ziel

Das Modell unterstützt:

* langfristige Planung
* Priorisierung von Maßnahmen
* Anpassung an den Klimawandel

---

##  Autor

Samuel Schweizer
Projekt Baumversorgung
