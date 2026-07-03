import React, { useMemo, useRef, useState } from "react";
import html2canvas from "html2canvas";
import { ComposedChart, Area, Line, XAxis, YAxis, ReferenceLine, ReferenceDot, Tooltip, ResponsiveContainer, CartesianGrid, Bar } from "recharts";

/* ─────────────────────────────────────────────────────────────
   Winterthur Stadtbaum-Modell v7 — interaktives Dashboard
   Repliziert die Engine aus winterthur_tree_stochastic_goal_planning_v7.py:
   p_fail = clip(base_p(Alter) · stress · (1+trend) · site^1 · mgmt^0.6 · life_mult, 1e-4, 0.50)
   base_p = empirisch kalibrierte Alters-Hazard aus dem Kataster (Sterbetafel,
   Laplace α=0.5, min_risk_set=20, clip≤0.35). Neue Bäume starten mit Alter 10.
   ───────────────────────────────────────────────────────────── */

/* Winterthur-Identität: Schwarz-Rot auf Weiss (Stadtlogo), heraldisches Rot, Schweizer Raster */
const PINE="#ece9e1",PANEL="#ffffff",CARD="#ffffff",LINE_GRID="#d9d7cf";
const PAPER="#1c1c1a",SAGE="#6f6d66",MOSS="#e1000f",MOSS_DIM="#b51020";
const AMBER="#c8860d",BARK="#9a9890",BRICK="#b00020",TEXT_MUTE="#7a786f";
const GREEN="#2f7d4f";
const CURRENT_YEAR=2026,YEARS=100,MAXAGE=200,MILESTONES=[4,25,50,100];

/* Echte kalibrierte Alters-Hazard-Kurve (Index = Alter, aus Kataster 2026) */
const HAZARD=[0.00136,0.00238,0.00443,0.00581,0.00651,0.00489,0.00659,0.00482,0.00448,0.0065,0.00457,0.00865,0.00868,0.00454,0.00495,0.00626,0.00629,0.00477,0.00548,0.00632,0.00696,0.00641,0.00757,0.00548,0.00757,0.00466,0.00666,0.00732,0.00787,0.01058,0.01247,0.00691,0.00887,0.0072,0.01138,0.01162,0.01116,0.01002,0.00865,0.01005,0.0141,0.00958,0.01369,0.00938,0.00864,0.00923,0.01028,0.01451,0.01048,0.01583,0.01571,0.01248,0.01266,0.0127,0.01326,0.01017,0.01059,0.01388,0.01328,0.00888,0.01312,0.0113,0.02701,0.01075,0.01403,0.01136,0.00874,0.00972,0.01031,0.00672,0.01377,0.00798,0.01003,0.0107,0.0085,0.01178,0.01277,0.00899,0.00508,0.01437,0.01654,0.01495,0.00816,0.00928,0.00655,0.01255,0.00792,0.0062,0.00867,0.00841,0.01725,0.00886,0.00927,0.00672,0.00319,0.00963,0.00923,0.00624,0.01123,0.00927,0.0054,0.01355,0.00467,0.00722,0.01234,0.00674,0.00619,0.00615,0.00694,0.0131,0.01103,0.00581,0.00668,0.01168,0.01019,0.00702,0.00624,0.01197,0.00583,0.00483,0.00812,0.01304,0.00523,0.0041,0.00648,0.00178,0.01029,0.00198,0.01525,0.00482,0.01042,0.00492,0.01686,0.01274,0.00836,0.01767,0.00235,0.01402,0.02376,0.00877,0.00884,0.00495,0.00497,0.007,0.00706,0.00507,0.00714,0.00598,0.0012,0.01095,0.01351,0.01122,0.00897,0.00646,0.00649,0.01197,0.00672,0.00177,0.01311,0.00189,0.0407,0.01008,0.35,0.02119,0.01327,0.00446,0.03125,0.01389,0.07944,0.01648,0.00556,0.00556,0.00562,0.00568,0.00568,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705];
const AGE_HIST=[75,270,333,724,649,479,222,304,218,205,186,258,542,243,279,301,543,327,395,243,407,285,189,150,88,177,209,116,146,194,120,240,95,71,195,199,396,136,97,44,250,393,39,45,75,56,276,64,57,66,148,178,90,64,129,11,461,10,148,27,139,138,24,31,35,29,471,18,127,40,50,105,18,14,101,9,321,8,65,9,18,122,2,4,59,0,198,6,20,13,51,85,36,110,5,2,148,5,27,6,15,108,6,2,3,14,146,7,24,4,11,10,25,0,5,1,239,6,7,4,36,10,3,2,2,15,60,4,17,3,1,28,4,1,2,2,105,2,1,0,0,0,1,1,0,1,71,0,5,0];
const START_ACTIVE=16621, AVG_HAZARD_TODAY=0.0081, LIFE_REF=130;

/* Pflanzstrategie: Näherung des Speziesauswahl-Effekts als Hazard-Faktor neuer Bäume */
const STRAT_NEW={ same_mix:1.0, long_life:0.90, climate_fit:0.93, balanced:0.91 };
const STRAT_LABEL={ same_mix:"Status quo", long_life:"Langlebig", climate_fit:"Klima-Fit", balanced:"Ausgewogen" };

/* Bestandsgewichteter climate_mult aus TreeGOER-Exceedance (future bio05 vs bio05_q-Grenze der Art),
   exp(-tree_k·max(0,future−grenze)), 1/clip(.,0.2,2.0), gemittelt über aktive Bäume. 45 % Artabdeckung. */
const CLIMATE={
  ssp126:{ q95:{0.08:1.000,0.15:1.000}, qrt3:{0.08:1.013,0.15:1.026} },
  ssp370:{ q95:{0.08:1.001,0.15:1.001}, qrt3:{0.08:1.015,0.15:1.030} },
  ssp585:{ q95:{0.08:1.033,0.15:1.076}, qrt3:{0.08:1.095,0.15:1.229} },
};
const SCEN_LABEL={ ssp126:"SSP1-2.6 · 26.9°C", ssp370:"SSP3-7.0 · 27.1°C", ssp585:"SSP5-8.5 · 32.2°C" };

/* Katasterbasierte Baumarten-Zusammensetzung für die Dashboard-Ansicht.
   Generiert aus 2026-04-13_Baumkataster_gesamte_Daten.csv und species_life_ranges.csv.
   count = aktive Bäume 2026 nach normalisierter Baumart; fallen = historische Fällungen.
   risk kombiniert historischen Fällanteil und Lebensdauer-Proxy. climateFit bleibt ein Proxy,
   solange CityTrees-II/TreeGOER-Artwerte nicht eingebettet sind. */
const SPECIES_BASE=[
  {"name": "Tilia europaea", "label": "Linde / Holländische Linde", "count": 1414, "fallen": 358, "avgAge": 43.3, "lifeMin": 60, "life": 180, "lifeMax": 300, "climateFit": 0.66, "risk": 0.66},
  {"name": "Acer campestre", "label": "Feldahorn", "count": 1266, "fallen": 350, "avgAge": 33.6, "lifeMin": 50, "life": 125, "lifeMax": 200, "climateFit": 0.64, "risk": 0.82},
  {"name": "Quercus robur", "label": "Stieleiche", "count": 1185, "fallen": 221, "avgAge": 39.4, "lifeMin": 60, "life": 180, "lifeMax": 300, "climateFit": 0.7, "risk": 0.58},
  {"name": "Carpinus betulus", "label": "Hainbuche", "count": 1074, "fallen": 316, "avgAge": 34.4, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.66, "risk": 0.77},
  {"name": "Acer platanoides", "label": "Spitzahorn", "count": 917, "fallen": 473, "avgAge": 33.7, "lifeMin": 50, "life": 125, "lifeMax": 200, "climateFit": 0.56, "risk": 1.03},
  {"name": "Betula pendula", "label": "Birke", "count": 840, "fallen": 588, "avgAge": 28.3, "lifeMin": 45, "life": 60, "lifeMax": 75, "climateFit": 0.42, "risk": 1.63},
  {"name": "Fagus sylvatica", "label": "Rotbuche", "count": 781, "fallen": 355, "avgAge": 61.4, "lifeMin": 60, "life": 155, "lifeMax": 250, "climateFit": 0.45, "risk": 0.89},
  {"name": "Pinus sylvatica", "label": "Wald-Föhre", "count": 701, "fallen": 340, "avgAge": 42.2, "lifeMin": 80, "life": 120, "lifeMax": 150, "climateFit": 0.58, "risk": 1.03},
  {"name": "Platanus x acerifolia", "label": "Platane", "count": 561, "fallen": 78, "avgAge": 58.1, "lifeMin": 165, "life": 220, "lifeMax": 275, "climateFit": 0.78, "risk": 0.47},
  {"name": "Aesculus hippocastanum", "label": "Rosskastanie", "count": 507, "fallen": 135, "avgAge": 70.1, "lifeMin": 60, "life": 105, "lifeMax": 150, "climateFit": 0.44, "risk": 0.88},
  {"name": "Acer pseudoplatanus", "label": "Bergahorn", "count": 487, "fallen": 443, "avgAge": 42.3, "lifeMin": 50, "life": 125, "lifeMax": 200, "climateFit": 0.48, "risk": 1.22},
  {"name": "Tilia platyphyllos", "label": "Sommerlinde", "count": 449, "fallen": 125, "avgAge": 61.5, "lifeMin": 60, "life": 180, "lifeMax": 300, "climateFit": 0.68, "risk": 0.69},
  {"name": "Tilia cordata", "label": "Winterlinde", "count": 426, "fallen": 77, "avgAge": 20.3, "lifeMin": 60, "life": 180, "lifeMax": 300, "climateFit": 0.7, "risk": 0.58},
  {"name": "Prunus avium", "label": "Vogelkirsche", "count": 300, "fallen": 73, "avgAge": 18.9, "lifeMin": 60, "life": 80, "lifeMax": 100, "climateFit": 0.54, "risk": 0.98},
  {"name": "Prunus serrulata", "label": "Zierkirsche", "count": 264, "fallen": 294, "avgAge": 28.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.45, "risk": 1.26},
  {"name": "Quercus cerris", "label": "Zerreiche", "count": 261, "fallen": 34, "avgAge": 8.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.76, "risk": 0.59},
  {"name": "Taxus baccata", "label": "Eibe", "count": 257, "fallen": 152, "avgAge": 75.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.05},
  {"name": "Fraxinus excelsior", "label": "Esche", "count": 224, "fallen": 198, "avgAge": 44.8, "lifeMin": 90, "life": 120, "lifeMax": 150, "climateFit": 0.38, "risk": 1.23},
  {"name": "Robinia pseudoacacia", "label": "Robinie", "count": 210, "fallen": 342, "avgAge": 32.1, "lifeMin": 90, "life": 120, "lifeMax": 150, "climateFit": 0.68, "risk": 1.42},
  {"name": "Juglans regia", "label": "Walnuss", "count": 199, "fallen": 95, "avgAge": 24.6, "lifeMin": 120, "life": 160, "lifeMax": 200, "climateFit": 0.64, "risk": 0.89},
  {"name": "Pinus nigra", "label": "Schwarz-Föhre", "count": 167, "fallen": 80, "avgAge": 75.6, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.66, "risk": 0.92},
  {"name": "Picea abies", "label": "Fichte", "count": 162, "fallen": 339, "avgAge": 69.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.35, "risk": 1.42},
  {"name": "Gleditsia triacanthos", "label": "Lederhülsenbaum", "count": 149, "fallen": 15, "avgAge": 18.4, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.86, "risk": 0.49},
  {"name": "Fraxinus ornus", "label": "Manna-. Blumenesche", "count": 129, "fallen": 22, "avgAge": 22.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.67},
  {"name": "Malus Hybride", "label": "Zierapfel", "count": 129, "fallen": 155, "avgAge": 29.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.44, "risk": 1.28},
  {"name": "Corylus colurna", "label": "Baumhasel", "count": 128, "fallen": 79, "avgAge": 30.2, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.72, "risk": 1.0},
  {"name": "Alnus glutinosa", "label": "Schwarzerle", "count": 124, "fallen": 47, "avgAge": 18.0, "lifeMin": 50, "life": 60, "lifeMax": 80, "climateFit": 0.45, "risk": 1.34},
  {"name": "Aesculus x carnea", "label": "Rotblühende Rosskastanie. gefüllte", "count": 105, "fallen": 27, "avgAge": 34.9, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.79},
  {"name": "Populus nigra", "label": "Schwarzpappel", "count": 105, "fallen": 96, "avgAge": 34.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.47, "risk": 1.2},
  {"name": "Salix alba", "label": "Silberweide", "count": 105, "fallen": 140, "avgAge": 25.4, "lifeMin": 64, "life": 85, "lifeMax": 106, "climateFit": 0.26, "risk": 1.62},
  {"name": "Prunus padus", "label": "Traubenkirsche", "count": 100, "fallen": 92, "avgAge": 21.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.47, "risk": 1.2},
  {"name": "Prunus subhirtella", "label": "Japanische Zierkirsche. Higan-Kirsche", "count": 99, "fallen": 8, "avgAge": 19.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.49},
  {"name": "Alnus x spaethii", "label": "Purpur - Erle", "count": 92, "fallen": 13, "avgAge": 12.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.62},
  {"name": "Sophora japonica", "label": "Schnurbaum", "count": 90, "fallen": 43, "avgAge": 25.9, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.84, "risk": 0.92},
  {"name": "Malus domestica", "label": "Apfel", "count": 79, "fallen": 4, "avgAge": 8.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.62, "risk": 0.45},
  {"name": "Parrotia persica", "label": "Eisenholzbaum. Parrotie", "count": 79, "fallen": 16, "avgAge": 36.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.72},
  {"name": "Cercidiphyllum japonicum", "label": "Katsurabaum. Japanischer Kuchenbaum", "count": 59, "fallen": 9, "avgAge": 37.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.64},
  {"name": "Prunus", "label": "Frühe Zierkirsche", "count": 59, "fallen": 24, "avgAge": 14.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.54, "risk": 0.93},
  {"name": "Liquidambar styraciflua", "label": "Amberbaum", "count": 58, "fallen": 9, "avgAge": 16.7, "lifeMin": 112, "life": 150, "lifeMax": 188, "climateFit": 0.74, "risk": 0.6},
  {"name": "Liriodendron tulipifera", "label": "Tulpenbaum", "count": 58, "fallen": 13, "avgAge": 48.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.75},
  {"name": "Celtis australis", "label": "Europäischer Zürgelbaum", "count": 57, "fallen": 7, "avgAge": 19.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.59},
  {"name": "Amelanchier lamarckii", "label": "Kupfer-Felsenbirne", "count": 54, "fallen": 38, "avgAge": 17.7, "lifeMin": 45, "life": 60, "lifeMax": 75, "climateFit": 0.25, "risk": 1.64},
  {"name": "Prunus domestica", "label": "Kultur-Pflaume. Zwetschge", "count": 54, "fallen": 19, "avgAge": 12.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.89},
  {"name": "Quercus rubra", "label": "Amerikanische Roteiche", "count": 54, "fallen": 40, "avgAge": 32.0, "lifeMin": 60, "life": 180, "lifeMax": 300, "climateFit": 0.69, "risk": 0.96},
  {"name": "Acer saccharinum", "label": "Silberahorn", "count": 53, "fallen": 82, "avgAge": 62.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.35},
  {"name": "Prunus sargentii", "label": "Scharlach-Kirsche", "count": 53, "fallen": 3, "avgAge": 4.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.62, "risk": 0.45},
  {"name": "Sorbus aria", "label": "Echte Mehlbeere", "count": 48, "fallen": 22, "avgAge": 21.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.53, "risk": 0.97},
  {"name": "0_UNBESTIMMT wird nachgetragen", "label": "0_UNBESTIMMT", "count": 46, "fallen": 14, "avgAge": 25.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.84},
  {"name": "Alnus incana", "label": "Grau-Erle. Weiss-Erle", "count": 45, "fallen": 112, "avgAge": 18.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.38, "risk": 1.46},
  {"name": "Tilia tomentosa", "label": "Silber-Linde", "count": 45, "fallen": 78, "avgAge": 70.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.41, "risk": 1.38},
  {"name": "Fraxinus excelsiors Glorie'", "label": "Esche Westhof's Glorie", "count": 44, "fallen": 20, "avgAge": 18.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.53, "risk": 0.97},
  {"name": "Larix decidua", "label": "Europäische Lärche", "count": 43, "fallen": 35, "avgAge": 44.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.48, "risk": 1.16},
  {"name": "Pyrus communis", "label": "Birnbaum", "count": 43, "fallen": 20, "avgAge": 13.5, "lifeMin": 60, "life": 80, "lifeMax": 100, "climateFit": 0.33, "risk": 1.25},
  {"name": "Koelreuteria paniculata", "label": "Blasenbaum. Blasenesche", "count": 42, "fallen": 7, "avgAge": 20.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.67},
  {"name": "Populus tremula", "label": "Espe. Zitterpappel", "count": 42, "fallen": 41, "avgAge": 18.3, "lifeMin": 70, "life": 85, "lifeMax": 100, "climateFit": 0.28, "risk": 1.5},
  {"name": "Acer palmatum", "label": "Japanischer Ahorn. Fächer-Ahorn", "count": 40, "fallen": 4, "avgAge": 55.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.55},
  {"name": "Amelanchier arborea", "label": "Schnee-Felsenbirne", "count": 40, "fallen": 1, "avgAge": 7.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.63, "risk": 0.45},
  {"name": "Sorbus intermedia", "label": "Schwedische Mehlbeere. Oxelbeere", "count": 38, "fallen": 22, "avgAge": 12.2, "lifeMin": 90, "life": 120, "lifeMax": 150, "climateFit": 0.47, "risk": 1.09},
  {"name": "Acer cappadocicum", "label": "Kolchischer Ahorn", "count": 35, "fallen": 12, "avgAge": 49.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.88},
  {"name": "Pyrus calleryana", "label": "Chinesische Wild-Birne", "count": 35, "fallen": 10, "avgAge": 20.5, "lifeMin": 52, "life": 70, "lifeMax": 88, "climateFit": 0.32, "risk": 1.13},
  {"name": "Betula utilis", "label": "Himalaja-Birke", "count": 34, "fallen": 6, "avgAge": 54.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.69},
  {"name": "Ulmus", "label": "Lobel-Ulme", "count": 34, "fallen": 3, "avgAge": 7.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.52},
  {"name": "Celtis occidentalis", "label": "Amerikanischer Zürgelbaum", "count": 33, "fallen": 7, "avgAge": 48.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.74},
  {"name": "Cornus mas", "label": "Kornelkirsche. Tierlibaum", "count": 33, "fallen": 12, "avgAge": 55.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.54, "risk": 0.9},
  {"name": "Ostrya carpinifolia", "label": "Europäische Hopfenbuche", "count": 33, "fallen": 0, "avgAge": 4.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.63, "risk": 0.45},
  {"name": "Quercus petraea", "label": "Traubeneiche", "count": 33, "fallen": 16, "avgAge": 5.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.52, "risk": 0.99},
  {"name": "Chamaecyparis lawsoniana", "label": "Lawsons Scheinzypresse", "count": 32, "fallen": 17, "avgAge": 104.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.02},
  {"name": "Cladrastis kentukea", "label": "Amerikanisches Gelbholz", "count": 32, "fallen": 0, "avgAge": 7.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.63, "risk": 0.45},
  {"name": "Sequoiadendron giganteum", "label": "Riesenmammutbaum", "count": 31, "fallen": 1, "avgAge": 129.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.62, "risk": 0.45},
  {"name": "Sorbus aucuparia", "label": "Eberesche. Vogelbeere", "count": 31, "fallen": 99, "avgAge": 17.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.37, "risk": 1.51},
  {"name": "Crataegus x lavallei", "label": "Lederblättriger Weissdorn", "count": 30, "fallen": 258, "avgAge": 26.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.32, "risk": 1.64},
  {"name": "Castanea sativa", "label": "Esskastanie. Edelkastanie", "count": 27, "fallen": 3, "avgAge": 5.6, "lifeMin": 70, "life": 135, "lifeMax": 200, "climateFit": 0.62, "risk": 0.57},
  {"name": "Crataegus monogyna", "label": "Säulen Weissdorn", "count": 25, "fallen": 20, "avgAge": 54.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.48, "risk": 1.15},
  {"name": "Salix caprea", "label": "Salweide", "count": 25, "fallen": 56, "avgAge": 29.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.44},
  {"name": "Salix fragilis", "label": "Knack-Weide. Bruch-Weide", "count": 25, "fallen": 3, "avgAge": 17.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.6},
  {"name": "Sorbus domestica", "label": "Speierling", "count": 23, "fallen": 20, "avgAge": 23.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.47, "risk": 1.18},
  {"name": "Catalpa bignonioides", "label": "Trompetenbaum", "count": 22, "fallen": 12, "avgAge": 33.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.03},
  {"name": "Ilex aquifolium", "label": "Europäische Stechpalme", "count": 22, "fallen": 15, "avgAge": 69.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.1},
  {"name": "Fraxinus angustifolia", "label": "Schmalblättrige Esche", "count": 21, "fallen": 0, "avgAge": 21.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.63, "risk": 0.45},
  {"name": "Juglans nigra", "label": "Schwarznussbaum", "count": 21, "fallen": 11, "avgAge": 60.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.02},
  {"name": "Acer negundo", "label": "Eschen-Ahorn", "count": 20, "fallen": 32, "avgAge": 40.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.35},
  {"name": "Alnus cordata", "label": "Italienische Erle. Herzblättrige Erle", "count": 20, "fallen": 18, "avgAge": 36.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.47, "risk": 1.19},
  {"name": "Gingko biloba", "label": "Gingko. Mädchenhaarbaum", "count": 20, "fallen": 6, "avgAge": 52.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.85},
  {"name": "Morus alba", "label": "Weisse Maulbeere", "count": 20, "fallen": 21, "avgAge": 31.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.24},
  {"name": "Sorbus torminalis", "label": "Elsbeere", "count": 20, "fallen": 10, "avgAge": 8.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.52, "risk": 1.01},
  {"name": "Tsuga canadensis", "label": "Kanadische Hemlockstanne", "count": 19, "fallen": 13, "avgAge": 61.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.11},
  {"name": "Acer monspessulanum", "label": "Felsenahorn. Französischer Ahorn", "count": 18, "fallen": 2, "avgAge": 15.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.6},
  {"name": "Abies nordmanniana", "label": "Nordmann-Tanne", "count": 17, "fallen": 3, "avgAge": 10.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Carya illinoinensis", "label": "Pekannuss", "count": 17, "fallen": 1, "avgAge": 10.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.49},
  {"name": "Larix kaempferi", "label": "Japanische Lärche", "count": 17, "fallen": 25, "avgAge": 38.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.43, "risk": 1.33},
  {"name": "Aesculus hybrida", "label": "Rosskastanie", "count": 16, "fallen": 13, "avgAge": 89.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.48, "risk": 1.16},
  {"name": "Magnolia kobus", "label": "Kobushi Magnolie", "count": 16, "fallen": 2, "avgAge": 13.9, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.63},
  {"name": "Picea omorika", "label": "Serbische Fichte", "count": 16, "fallen": 48, "avgAge": 56.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.37, "risk": 1.49},
  {"name": "Pinus nigra ssp. nigra", "label": "Österreichische Scharzkiefer", "count": 16, "fallen": 4, "avgAge": 60.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.56, "risk": 0.8},
  {"name": "Ulmus glabra", "label": "Berg-Ulme", "count": 16, "fallen": 8, "avgAge": 50.2, "lifeMin": 98, "life": 130, "lifeMax": 162, "climateFit": 0.52, "risk": 1.01},
  {"name": "Acer opalus Mill.", "label": "Schneeballblättriger Ahorn", "count": 15, "fallen": 5, "avgAge": 2.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.88},
  {"name": "Magnolia x soulangeana", "label": "Tulpen-Mangolie", "count": 15, "fallen": 3, "avgAge": 56.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.74},
  {"name": "Pterocarya fraxinifolia", "label": "Kaukasische Flügelnuss", "count": 15, "fallen": 11, "avgAge": 68.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.13},
  {"name": "Populus alba", "label": "Silber-Pappel", "count": 14, "fallen": 4, "avgAge": 32.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.56, "risk": 0.84},
  {"name": "Quercus palustris", "label": "Sumpf-Eiche", "count": 14, "fallen": 3, "avgAge": 8.4, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.76},
  {"name": "Prunus cerasifera", "label": "Blut-Pflaume", "count": 13, "fallen": 30, "avgAge": 30.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.44},
  {"name": "Pyrus communis var. sativa", "label": "Kultursorten", "count": 12, "fallen": 25, "avgAge": 43.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.4, "risk": 1.42},
  {"name": "Aesculus flava", "label": "Gelbe Pavie. Gelbe Rosskastanie", "count": 11, "fallen": 4, "avgAge": 32.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.54, "risk": 0.92},
  {"name": "Fraxinus pennsylvanica", "label": "Rotesche Cimmaron", "count": 11, "fallen": 0, "avgAge": 2.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.63, "risk": 0.45},
  {"name": "Quercus ilex", "label": "Steineiche", "count": 11, "fallen": 1, "avgAge": 3.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.59},
  {"name": "Thuja plicata", "label": "Riesen-Lebensbaum", "count": 11, "fallen": 8, "avgAge": 118.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.13},
  {"name": "Crataegus laevigata", "label": "Zweigriffeliger Weissdorn", "count": 10, "fallen": 42, "avgAge": 26.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.35, "risk": 1.55},
  {"name": "Tilia americana", "label": "Amerikanische Linde", "count": 10, "fallen": 0, "avgAge": 114.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.62, "risk": 0.45},
  {"name": "Ailanthus altissima", "label": "Götterbaum", "count": 9, "fallen": 10, "avgAge": 57.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.45, "risk": 1.25},
  {"name": "Amelanchier canadensis", "label": "Kanadische Felsenbirne", "count": 9, "fallen": 2, "avgAge": 24.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.79},
  {"name": "Cercis siliquastrum", "label": "Gewöhnlicher Judasbaum", "count": 9, "fallen": 1, "avgAge": 8.9, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.59, "risk": 0.64},
  {"name": "Davidia involucrata", "label": "Taubenbaum. Taschentuchbaum", "count": 8, "fallen": 37, "avgAge": 43.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.35, "risk": 1.56},
  {"name": "Gymnocladus dioicus", "label": "Geweihbaum", "count": 8, "fallen": 5, "avgAge": 10.1, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.5, "risk": 1.08},
  {"name": "Salix x sepulcralis", "label": "Echte Trauer-Weide", "count": 8, "fallen": 2, "avgAge": 6.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.56, "risk": 0.82},
  {"name": "Toona sinensis", "label": "Surenbaum. Chinesischer Gemüsebaum", "count": 8, "fallen": 2, "avgAge": 40.9, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.56, "risk": 0.82},
  {"name": "Abies alba", "label": "Weisstanne", "count": 7, "fallen": 15, "avgAge": 63.6, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.4, "risk": 1.42},
  {"name": "Acer ginnala", "label": "Feuer-Ahorn", "count": 7, "fallen": 36, "avgAge": 71.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.34, "risk": 1.57},
  {"name": "Prunus persica", "label": "Pfirsich", "count": 7, "fallen": 0, "avgAge": 0.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.62, "risk": 0.45},
  {"name": "Ulmus hollandica", "label": "Holländische Ulme", "count": 7, "fallen": 2, "avgAge": 5.0, "lifeMin": 98, "life": 130, "lifeMax": 162, "climateFit": 0.55, "risk": 0.86},
  {"name": "Ulmus minor", "label": "Feldulme", "count": 7, "fallen": 7, "avgAge": 63.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Acer rubrum", "label": "Rot-Ahorn", "count": 6, "fallen": 1, "avgAge": 4.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.75},
  {"name": "Acer x", "label": "Roter Feld-Ahorn", "count": 6, "fallen": 1, "avgAge": 57.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.75},
  {"name": "Magnolia", "label": "Tulpen-Magnolie Heaven Scent", "count": 6, "fallen": 0, "avgAge": 16.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.46},
  {"name": "Quercus macranthera", "label": "Persische Eiche", "count": 6, "fallen": 0, "avgAge": 57.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.46},
  {"name": "Tilia x moltkei", "label": "Moltke-Linde", "count": 6, "fallen": 2, "avgAge": 89.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.54, "risk": 0.91},
  {"name": "Ulmus New Horizon", "label": "Ulme Resistent", "count": 6, "fallen": 1, "avgAge": 7.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.57, "risk": 0.75},
  {"name": "Ulmus laevis", "label": "Flatter-Ulme", "count": 6, "fallen": 2, "avgAge": 19.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.54, "risk": 0.91},
  {"name": "Acer freemanii \"Autumn Blaze\"", "label": "Flammen Ahorn", "count": 5, "fallen": 0, "avgAge": 1.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.5},
  {"name": "Cornus florida", "label": "Blüten-Hartriegel", "count": 5, "fallen": 0, "avgAge": 24.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.5},
  {"name": "Populus x canescens", "label": "Grau-Pappel", "count": 5, "fallen": 12, "avgAge": 45.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.44},
  {"name": "Quercus frainetto", "label": "Ungarische Eiche", "count": 5, "fallen": 0, "avgAge": 22.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.5},
  {"name": "Quercus pubescens", "label": "Flaumeiche", "count": 5, "fallen": 0, "avgAge": 3.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.61, "risk": 0.5},
  {"name": "Acer griseum", "label": "Zimt-Ahorn", "count": 4, "fallen": 0, "avgAge": 13.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.55},
  {"name": "Betula nigra", "label": "Schwarz-Birke", "count": 4, "fallen": 2, "avgAge": 30.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.03},
  {"name": "Betula papyrifera", "label": "Papier-Birke", "count": 4, "fallen": 3, "avgAge": 26.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.48, "risk": 1.14},
  {"name": "Cedrus atlantica", "label": "Atlaszeder", "count": 4, "fallen": 0, "avgAge": 57.2, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.55},
  {"name": "Cydonia oblonga", "label": "Echte Quitte", "count": 4, "fallen": 2, "avgAge": 11.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.03},
  {"name": "Eco Arbore", "label": "Öko-Baum", "count": 4, "fallen": 0, "avgAge": 81.8, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.55},
  {"name": "Malus tschonoskii", "label": "Scharlach Apfel", "count": 4, "fallen": 0, "avgAge": 1.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.55},
  {"name": "Phellodendron amurense", "label": "Amur-Korkbaum", "count": 4, "fallen": 1, "avgAge": 28.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Acer freemanii", "label": "Flammenahorn", "count": 3, "fallen": 0, "avgAge": 0.7, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Acer saccharum", "label": "Zuckerahorn", "count": 3, "fallen": 3, "avgAge": 60.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Betula albosinensis", "label": "Rote China-Birke", "count": 3, "fallen": 0, "avgAge": 23.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Buxus sempervirens", "label": "Gewöhnlicher Buchsbaum", "count": 3, "fallen": 2, "avgAge": 70.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.12},
  {"name": "Catalpa speciosa", "label": "Prächtiger Trompetenbaum", "count": 3, "fallen": 0, "avgAge": 25.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Magnolia grandiflora", "label": "Immergrüne Magnolie", "count": 3, "fallen": 0, "avgAge": 41.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Metasequoia glyptostroboides", "label": "Urwelt-Mammutbaum", "count": 3, "fallen": 2, "avgAge": 4.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.12},
  {"name": "Picea orientalis", "label": "Kaukasus-Fichte", "count": 3, "fallen": 8, "avgAge": 97.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.46},
  {"name": "Prunus fruticosa", "label": "Steppenkirsche. Zwerg-Kirsche", "count": 3, "fallen": 0, "avgAge": 18.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Prunus umineko", "label": "Japanische Säulenkirsche", "count": 3, "fallen": 2, "avgAge": 14.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.49, "risk": 1.12},
  {"name": "Pseudotsuga menziesii", "label": "Gewöhnliche Douglasie", "count": 3, "fallen": 4, "avgAge": 59.3, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.44, "risk": 1.3},
  {"name": "Sorbus americana", "label": "Amerikanische Eberesche", "count": 3, "fallen": 0, "avgAge": 18.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Thuja occidentalis", "label": "Abendländischer Lebensbaum", "count": 3, "fallen": 8, "avgAge": 68.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.46},
  {"name": "Ulmus rebona", "label": "Rebona-Ulme", "count": 3, "fallen": 0, "avgAge": 8.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.6, "risk": 0.61},
  {"name": "Acer tataricum", "label": "Tatarischer Steppen-Ahorn", "count": 2, "fallen": 5, "avgAge": 26.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.43},
  {"name": "Chamaecyparis pisifera", "label": "Kegelförmige Sawara Scheinzypresse", "count": 2, "fallen": 6, "avgAge": 82.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.38, "risk": 1.47},
  {"name": "Cupressus arizonica", "label": "Blaue Arizona-Zypresse", "count": 2, "fallen": 4, "avgAge": 119.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.41, "risk": 1.39},
  {"name": "Elaeagnus angustifolia", "label": "Schmalblättrige Ölweide", "count": 2, "fallen": 8, "avgAge": 14.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.36, "risk": 1.52},
  {"name": "Fagus sylvatica cuprea/purpurea", "label": "Blutbuche", "count": 2, "fallen": 1, "avgAge": 152.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.51, "risk": 1.06},
  {"name": "Fraxinus americana", "label": "Weiss-Esche", "count": 2, "fallen": 5, "avgAge": 47.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.43},
  {"name": "Juglans mandshurica Maxim", "label": "Mandschurische Walnuss", "count": 2, "fallen": 0, "avgAge": 131.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Laburnum vulgare", "label": "Einheimischer Goldregen", "count": 2, "fallen": 6, "avgAge": 37.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.38, "risk": 1.47},
  {"name": "Mespilus germanica", "label": "Mispel", "count": 2, "fallen": 3, "avgAge": 36.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.43, "risk": 1.32},
  {"name": "Morus nigra", "label": "Schwarze Maulbeere", "count": 2, "fallen": 5, "avgAge": 53.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.43},
  {"name": "Nyssa sylvatica", "label": "Wald-Tupelobaum", "count": 2, "fallen": 0, "avgAge": 11.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Paulownia tomentosa", "label": "Blauglockenbaum", "count": 2, "fallen": 7, "avgAge": 19.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.37, "risk": 1.5},
  {"name": "Picea abies Ohlendorffii'", "label": "Kegelfichte", "count": 2, "fallen": 0, "avgAge": 66.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Picea pungens", "label": "Stech-Fichte", "count": 2, "fallen": 4, "avgAge": 38.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.41, "risk": 1.39},
  {"name": "Prunus insititia", "label": "Kriechende Haferschlehe. Kriechen-Pflaume", "count": 2, "fallen": 5, "avgAge": 63.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.43},
  {"name": "Prunus mahaleb", "label": "Steinweichsel", "count": 2, "fallen": 0, "avgAge": 3.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Prunus serotina", "label": "Amerikanische / Spätblühende Traubenkirsche", "count": 2, "fallen": 2, "avgAge": 21.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Quercus texana", "label": "Texanische Eiche", "count": 2, "fallen": 0, "avgAge": 0.5, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Unbekannt", "label": "Unbekannt", "count": 2, "fallen": 0, "avgAge": 66.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.58, "risk": 0.71},
  {"name": "Abies cephalonica", "label": "Griechische Tanne", "count": 1, "fallen": 0, "avgAge": 161.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Abies concolor", "label": "Kolorado-Tanne. Grau-Tanne", "count": 1, "fallen": 0, "avgAge": 76.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Abies pinsapo", "label": "Spanische Tanne", "count": 1, "fallen": 0, "avgAge": 46.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Acer carpinifolium", "label": "Hainbuchenblättriger Ahorn", "count": 1, "fallen": 0, "avgAge": 7.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Acer platanoides \"Allershausen\"", "label": "Spitzahorn Allershausen", "count": 1, "fallen": 0, "avgAge": 0.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Acer platanoides \"Norwegian Sunset\"", "label": "Acer platanoides \"Norwegian Sunset\"", "count": 1, "fallen": 0, "avgAge": 0.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Acer velutinum", "label": "Samt-Ahorn", "count": 1, "fallen": 0, "avgAge": 136.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Albizia julibrissin", "label": "Seidenbaum", "count": 1, "fallen": 0, "avgAge": 1.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Calocedrus decurrens", "label": "Weihrauchzeder", "count": 1, "fallen": 2, "avgAge": 206.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.37},
  {"name": "Cedrus libani", "label": "Libanon-Zeder", "count": 1, "fallen": 0, "avgAge": 7.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Chamaecyparis nootkatensis", "label": "Nootka Scheinzypresse", "count": 1, "fallen": 1, "avgAge": 84.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Chamaecyparis obtusa", "label": "Hinoki-Scheinzypresse", "count": 1, "fallen": 0, "avgAge": 65.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Chionanthus retusus", "label": "Chinesischer Schneeflockenstrauch", "count": 1, "fallen": 0, "avgAge": 16.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Cornus controversa", "label": "Etagen Hartriegel. Pagoden-Hartriegel", "count": 1, "fallen": 1, "avgAge": 50.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Crataegus coccinoides", "label": "Weissdorn", "count": 1, "fallen": 1, "avgAge": 75.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Crataegus crus-galli", "label": "Hahnensporn-Weissdorn", "count": 1, "fallen": 3, "avgAge": 50.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.45},
  {"name": "Eucommia ulmoides", "label": "Guttaperchabaum", "count": 1, "fallen": 0, "avgAge": 3.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Euonymus europaeus", "label": "Pfaffenhütchen. Gewöhnlicher Spindelstrauch", "count": 1, "fallen": 0, "avgAge": 18.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Fagus sylvatica Purpurea Pendula'", "label": "Schwarzrote Hänge-Buche", "count": 1, "fallen": 0, "avgAge": 35.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Larix x eurolepis", "label": "Hybridlärche", "count": 1, "fallen": 19, "avgAge": 56.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.31, "risk": 1.67},
  {"name": "Nothofagus antarctica", "label": "Antarktische Scheinbuche", "count": 1, "fallen": 10, "avgAge": 9.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.33, "risk": 1.62},
  {"name": "Pinus mugo", "label": "Bergföhre", "count": 1, "fallen": 5, "avgAge": 50.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.36, "risk": 1.53},
  {"name": "Pinus strobus", "label": "Weymouth-Kiefer", "count": 1, "fallen": 6, "avgAge": 6.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.35, "risk": 1.56},
  {"name": "Populus x", "label": "Bastard Scharzpappel", "count": 1, "fallen": 8, "avgAge": 66.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.33, "risk": 1.59},
  {"name": "Prunus armeniaca", "label": "Aprikose", "count": 1, "fallen": 0, "avgAge": 21.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Pterocarya stenoptera", "label": "Chinesische Flügelnuss", "count": 1, "fallen": 0, "avgAge": 72.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Quercus coccinea", "label": "Scharlach-Eiche", "count": 1, "fallen": 1, "avgAge": 3.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Rhamnus cathartica", "label": "Purgier-Kreuzdorn", "count": 1, "fallen": 3, "avgAge": 76.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.39, "risk": 1.45},
  {"name": "Robinia x ambigua", "label": "Robinie. Scheinakazie", "count": 1, "fallen": 1, "avgAge": 75.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Salix babylonica", "label": "Babylon-Weide. Echte Trauerweide", "count": 1, "fallen": 2, "avgAge": 67.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.37},
  {"name": "Salix cinerea", "label": "Asch-Weide", "count": 1, "fallen": 0, "avgAge": 15.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Salix matsudana", "label": "Korkenzieher-Weide", "count": 1, "fallen": 0, "avgAge": 27.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Sambucus nigra", "label": "Schwarzer Holunder", "count": 1, "fallen": 1, "avgAge": 36.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Sorbus x thuringiaca", "label": "Thüringische Mehlbeere", "count": 1, "fallen": 1, "avgAge": 5.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Taxodium distichum", "label": "Echte Sumpfzypresse", "count": 1, "fallen": 0, "avgAge": 7.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Thuja orientalis", "label": "Morgenländischer Lebensbaum", "count": 1, "fallen": 1, "avgAge": 27.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.46, "risk": 1.22},
  {"name": "Thujopsis dolabrata", "label": "Hiba-Lebensbaum", "count": 1, "fallen": 2, "avgAge": 54.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.37},
  {"name": "Tilia cordata `Böhlje`", "label": "Winterlinde `Böhlje", "count": 1, "fallen": 0, "avgAge": 8.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Tilia dasystyla", "label": "Kaukaische-Linde", "count": 1, "fallen": 0, "avgAge": 64.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.55, "risk": 0.86},
  {"name": "Zelkova serrata", "label": "Japanische Zelkove", "count": 1, "fallen": 2, "avgAge": 4.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.42, "risk": 1.37},
  {"name": "cornus sanguinea", "label": "Heimischer Hartriegel. Roter Hartriegel", "count": 1, "fallen": 10, "avgAge": 18.0, "lifeMin": 90, "life": 130, "lifeMax": 180, "climateFit": 0.33, "risk": 1.62},
];

const SPECIES_DATASET = {
  activeTotal: 16621,
  fallenTotal: 8351,
  plannedTotal: 494,
  speciesTotal: 214,
  lifeCoveredActive: 12678,
  lifeCoverage: 0.763,
  source: "Baumkataster 2026-04-13 + species_life_ranges.csv",
};

const SPECIES_NOTE="Katasterbasierte Artansicht: aktuelle Artanteile, historische Fällungen und mittleres Alter stammen aus dem Baumkataster 2026-04-13. Lebensdauerwerte stammen aus species_life_ranges.csv; Abdeckung der Lebensdauerwerte: 76.3 % der aktiven Bäume. Klima-Fit ist in dieser Dashboard-Version noch ein Proxy/Kommunikationswert, solange CityTrees-II und TreeGOER-Artwerte nicht direkt eingebettet sind.";

const DEFAULTS={
  N0:START_ACTIVE, scenario:"ssp126", treeQ:"q95", treeK:0.08, manualStress:1.0, siteFactor:1.0, mgmtFactor:1.0, climateTrendEnd:0.35,
  replacementRate:0.8, replacementDelay:2, annualNewTrees:300, newStart:1, newEnd:4, initAge:10,
  strategy:"balanced", lifeHazardWeight:0.5, lifeMode:"per_run",
  nRuns:100, targetCount:17000, seed:42, viewYearOff:24,
};
const PRESETS={
  konservativ:{annualNewTrees:250,strategy:"balanced",replacementRate:0.8},
  mittel:{annualNewTrees:300,strategy:"balanced",replacementRate:0.8},
  robust:{annualNewTrees:350,strategy:"balanced",replacementRate:0.8},
  maximal:{annualNewTrees:300,strategy:"balanced",replacementRate:1.0},
};

function mulberry32(a){return function(){a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function makeNormal(rng){let sp=null;return()=>{if(sp!==null){const s=sp;sp=null;return s;}let u,v,s;do{u=rng()*2-1;v=rng()*2-1;s=u*u+v*v;}while(s>=1||s===0);const m=Math.sqrt(-2*Math.log(s)/s);sp=v*m;return u*m;};}
const haz=a=>HAZARD[Math.min(a,MAXAGE-1)];

function startCohorts(N0){const arr=new Float64Array(MAXAGE);let s=0;for(let a=0;a<AGE_HIST.length&&a<MAXAGE;a++){arr[a]=AGE_HIST[a];s+=AGE_HIST[a];}const k=N0/s;for(let a=0;a<MAXAGE;a++)arr[a]*=k;return arr;}

function runCohort(p,stochastic,normal,m){
  const stratNew=STRAT_NEW[p.strategy];
  const climateMult=(CLIMATE[p.scenario]?.[p.treeQ]?.[p.treeK])??1.0;
  const stress=climateMult*p.manualStress;
  const old=startCohorts(p.N0), neu=new Float64Array(MAXAGE);
  const ia=Math.min(Math.max(0,p.initAge),MAXAGE-1);
  const deaths=new Float64Array(YEARS+1);
  const total=new Array(YEARS+1), classOld=[],classNew=[],medAge=[],young=[],old60=[];
  const snap=(oa,na)=>{const co=Array(18).fill(0),cn=Array(18).fill(0);let tot=0,y=0,o=0;
    for(let a=0;a<MAXAGE;a++){const c=Math.min(17,Math.floor(a/10));co[c]+=oa[a];cn[c]+=na[a];const t=oa[a]+na[a];tot+=t;if(a<20)y+=t;if(a>=60)o+=t;}
    classOld.push(co);classNew.push(cn);
    let acc=0,med=0;for(let a=0;a<MAXAGE;a++){acc+=oa[a]+na[a];if(acc>=tot/2){med=a;break;}}
    medAge.push(med);young.push(tot>0?y/tot:0);old60.push(tot>0?o/tot:0);return tot;};
  total[0]=snap(old,neu);
  for(let t=1;t<=YEARS;t++){
    const trend=p.climateTrendEnd*(t/YEARS);
    const rmO=m*stress*(1+trend)*Math.pow(p.siteFactor,1.0)*Math.pow(p.mgmtFactor,0.6);
    const rmN=rmO*stratNew;
    let dTot=0;const noO=new Float64Array(MAXAGE),noN=new Float64Array(MAXAGE);
    for(let a=MAXAGE-1;a>=0;a--){
      const bp=HAZARD[a];
      let pO=bp*rmO; if(pO<0.0001)pO=0.0001; if(pO>0.50)pO=0.50;
      let pN=bp*rmN; if(pN<0.0001)pN=0.0001; if(pN>0.50)pN=0.50;
      let dO=old[a]*pO,dN=neu[a]*pN;
      if(stochastic){
        if(old[a]>0){const sd=Math.sqrt(old[a]*pO*(1-pO));dO=Math.max(0,Math.min(old[a],dO+sd*normal()));}
        if(neu[a]>0){const sd=Math.sqrt(neu[a]*pN*(1-pN));dN=Math.max(0,Math.min(neu[a],dN+sd*normal()));}
      }
      dTot+=dO+dN; if(a+1<MAXAGE){noO[a+1]=old[a]-dO;noN[a+1]=neu[a]-dN;}
    }
    deaths[t]=dTot;
    const repl=t-p.replacementDelay>=1?Math.round(p.replacementRate*deaths[t-p.replacementDelay]):0;
    const extra=(t>=p.newStart&&t<=p.newEnd)?p.annualNewTrees:0;
    noN[ia]+=repl+extra;
    for(let a=0;a<MAXAGE;a++){old[a]=noO[a];neu[a]=noN[a];}
    total[t]=snap(old,neu);
  }
  return {total,classOld,classNew,medAge,young,old60,deaths};
}

function simulate(p){
  const det=runCohort(p,false,null,1);
  const rng=mulberry32(p.seed>>>0),normal=makeNormal(rng);
  const runs=[];
  for(let r=0;r<p.nRuns;r++){
    let sd=0.06; if(p.lifeMode==="per_run") sd=Math.sqrt(sd*sd+Math.pow(0.10*p.lifeHazardWeight,2));
    let m=Math.exp(normal()*sd); m=Math.min(2.0,Math.max(0.5,m));
    runs.push(runCohort(p,true,normal,m).total);
  }
  const series=[];
  for(let t=0;t<=YEARS;t++){
    const v=runs.map(s=>s[t]).sort((a,b)=>a-b);
    const q=x=>v[Math.min(v.length-1,Math.round(x*(v.length-1)))];
    series.push({offset:t,year:CURRENT_YEAR+t,p05:q(.05),p25:q(.25),p50:q(.5),p75:q(.75),p95:q(.95),
      mean:v.reduce((a,b)=>a+b,0)/v.length,band90:[q(.05),q(.95)],band50:[q(.25),q(.75)]});
  }
  return {series,det};
}


function strategyWeight(sp,strategy){
  if(strategy==="same_mix") return 1;
  if(strategy==="long_life") return Math.pow(sp.life/LIFE_REF,1.4);
  if(strategy==="climate_fit") return Math.pow(Math.max(0.2,sp.climateFit),2.2);
  // balanced: Klima-Fit, Lebensdauer und leichte Diversitätsförderung kombiniert
  return Math.pow(Math.max(0.2,sp.climateFit),1.4)*Math.pow(sp.life/LIFE_REF,0.8)*Math.pow(1/(sp.count+250),0.12);
}
function simulateSpecies(p,det){
  const baseSum=SPECIES_BASE.reduce((a,b)=>a+b.count,0)||1;
  let counts=SPECIES_BASE.map(sp=>sp.count*p.N0/baseSum);
  const history=[counts.slice()];
  for(let t=1;t<=YEARS;t++){
    const lossTotal=det.deaths[t]||0;
    const lossWeights=counts.map((c,i)=>c*Math.max(0.2,SPECIES_BASE[i].risk)*(1+(1-SPECIES_BASE[i].climateFit)*p.climateTrendEnd*(t/YEARS)));
    const lwSum=lossWeights.reduce((a,b)=>a+b,0)||1;
    const after=counts.map((c,i)=>Math.max(0,c-lossTotal*lossWeights[i]/lwSum));
    const repl=t-p.replacementDelay>=1?Math.round(p.replacementRate*(det.deaths[t-p.replacementDelay]||0)):0;
    const extra=(t>=p.newStart&&t<=p.newEnd)?p.annualNewTrees:0;
    const plant=repl+extra;
    const plantWeights=SPECIES_BASE.map(sp=>strategyWeight(sp,p.strategy));
    const pwSum=plantWeights.reduce((a,b)=>a+b,0)||1;
    counts=after.map((c,i)=>c+plant*plantWeights[i]/pwSum);
    history.push(counts.slice());
  }
  const rows=SPECIES_BASE.map((sp,i)=>({
    ...sp,
    start:history[0][i],
    mid:history[Math.min(YEARS,25)][i],
    end:history[YEARS][i],
    selected:history[p.viewYearOff]?.[i]??history[YEARS][i],
  })).sort((a,b)=>b.selected-a.selected);
  const totalNow=rows.reduce((a,b)=>a+b.selected,0)||1;
  const totalStart=rows.reduce((a,b)=>a+b.start,0)||1;
  const top5Share=rows.slice(0,5).reduce((a,b)=>a+b.selected,0)/totalNow;
  const climateFitShare=rows.filter(r=>r.climateFit>=0.70).reduce((a,b)=>a+b.selected,0)/totalNow;
  const shannon=(()=>{let h=0; rows.forEach(r=>{const q=r.selected/totalNow;if(q>0)h-=q*Math.log(q);}); return h/Math.log(rows.length);})();
  const chart=rows.slice(0,10).map(r=>({
    art:r.label,
    heute:Math.round(r.start),
    jahr:Math.round(r.selected),
    delta:Math.round(r.selected-r.start),
    fit:Math.round(r.climateFit*100),
    avgAge:r.avgAge,
    life:r.life,
    fallen:r.fallen,
  }));
  const trend=rows.slice(0,8).map(r=>({
    art:r.label,
    heute:Math.round(r.start/totalStart*1000)/10,
    jahr:Math.round(r.selected/totalNow*1000)/10,
    delta:Math.round((r.selected/totalNow-r.start/totalStart)*1000)/10,
    avgAge:r.avgAge ?? "–",
    life:r.life ?? "–",
  }));
  return {rows,chart,trend,top5Share,climateFitShare,shannon};
}

const fmt=n=>Math.round(n).toLocaleString("de-CH").replace(/,/g,"\u2019");
const yearAt=o=>CURRENT_YEAR+o;
function verdict(pt,target){if(!pt)return{label:"—",color:TEXT_MUTE,note:""};
  if(pt.p05>=target)return{label:"Robust erreicht",color:GREEN,note:"95 % der Läufe über dem Ziel"};
  if(pt.p50>=target)return{label:"Wahrscheinlich erreicht",color:"#5a9e63",note:"in über der Hälfte der Läufe"};
  if(pt.mean>=target)return{label:"Knapp / im Mittel",color:AMBER,note:"Mittelwert über Ziel, Median darunter"};
  return{label:"Verfehlt",color:BRICK,note:"Ziel im Median nicht erreicht"};}
function ageColor(i){const r=i/17,g=[201,199,191],a=[74,72,67];return `rgb(${Math.round(g[0]+(a[0]-g[0])*r)},${Math.round(g[1]+(a[1]-g[1])*r)},${Math.round(g[2]+(a[2]-g[2])*r)})`;}

function Slider({label,hint,value,min,max,step,onChange,display}){
  return(<div style={{marginBottom:15}}>
    <div className="flex items-baseline justify-between" style={{marginBottom:5}}>
      <span style={{fontSize:13,color:PAPER}}>{label}</span>
      <span style={{fontSize:13,color:MOSS,fontFamily:"var(--mono)",fontWeight:600}}>{display??value}</span></div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))} style={{width:"100%",accentColor:MOSS,height:4,cursor:"pointer"}}/>
    {hint&&<div style={{fontSize:11,color:TEXT_MUTE,marginTop:3}}>{hint}</div>}</div>);
}
function Segmented({label,value,options,onChange}){
  return(<div style={{marginBottom:15}}>
    <div style={{fontSize:13,color:PAPER,marginBottom:6}}>{label}</div>
    <div className="flex flex-wrap" style={{gap:6}}>
      {options.map(o=>{const a=o.value===value;return(<button key={o.value} onClick={()=>onChange(o.value)}
        style={{fontSize:12,padding:"6px 11px",borderRadius:7,cursor:"pointer",border:`1px solid ${a?MOSS:LINE_GRID}`,
        background:a?"rgba(225,0,15,0.07)":"transparent",color:a?MOSS:SAGE,fontFamily:"var(--mono)"}}>{o.label}</button>);})}</div></div>);
}
function Section({title,children}){return(<div style={{marginBottom:20}}>
  <div style={{fontSize:11,letterSpacing:1.6,textTransform:"uppercase",color:MOSS_DIM,fontWeight:700,marginBottom:11,borderBottom:`1px solid ${LINE_GRID}`,paddingBottom:6}}>{title}</div>{children}</div>);}
function MilestoneCard({off,pt,target}){const ok=pt.p50>=target;
  return(<div style={{background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"13px 15px",flex:"1 1 0",minWidth:120}}>
    <div style={{fontSize:11,color:TEXT_MUTE,letterSpacing:1}}>{yearAt(off)}</div>
    <div style={{fontFamily:"var(--display)",fontSize:28,fontWeight:600,lineHeight:1.1,color:PAPER,fontVariantNumeric:"tabular-nums"}}>{fmt(pt.p50)}</div>
    <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}>{fmt(pt.p05)} – {fmt(pt.p95)}</div>
    <div style={{marginTop:7,fontSize:11,fontWeight:600,color:ok?GREEN:(off===4?BRICK:TEXT_MUTE)}}>{off===4?(ok?"Ziel erreicht":"unter Ziel"):(ok?"über Start":"unter Start")}</div></div>);}
function ChartTip({active,payload,target}){if(!active||!payload||!payload.length)return null;const d=payload[0].payload;
  return(<div style={{background:"#ffffff",border:`1px solid ${LINE_GRID}`,borderRadius:9,padding:"10px 12px",fontFamily:"var(--mono)",fontSize:12,color:PAPER}}>
    <div style={{color:MOSS,fontWeight:700,marginBottom:4}}>{d.year}</div><div>Median&nbsp;&nbsp;<b>{fmt(d.p50)}</b></div>
    <div style={{color:SAGE}}>50 %&nbsp;&nbsp;{fmt(d.p25)} – {fmt(d.p75)}</div>
    <div style={{color:TEXT_MUTE}}>90 %&nbsp;&nbsp;{fmt(d.p05)} – {fmt(d.p95)}</div>
    <div style={{color:AMBER,marginTop:3}}>Ziel&nbsp;&nbsp;{fmt(target)}</div></div>);}

function HazardSpark(){
  const W=260,Hh=46,N=151,mx=0.04;
  const pts=Array.from({length:N},(_,a)=>{const x=a/(N-1)*W;const y=Hh-Math.min(haz(a),mx)/mx*Hh;return `${x.toFixed(1)},${y.toFixed(1)}`;}).join(" ");
  return(<svg width="100%" viewBox={`0 0 ${W} ${Hh}`} preserveAspectRatio="none" style={{display:"block"}}>
    <polyline points={pts} fill="none" stroke={MOSS} strokeWidth="1.6"/>
    {[0,50,100,150].map(a=><line key={a} x1={a/(N-1)*W} y1="0" x2={a/(N-1)*W} y2={Hh} stroke={LINE_GRID} strokeDasharray="1 4"/>)}
  </svg>);
}
function Pyramid({classOld,classNew}){
  const maxV=Math.max(1,...classOld.map((v,i)=>v+classNew[i]));
  const labels=Array.from({length:18},(_,i)=>i===17?"170+":`${i*10}–${i*10+9}`);
  return(<div style={{display:"flex",flexDirection:"column-reverse",gap:3}}>
    {labels.map((lab,i)=>{const o=classOld[i],n=classNew[i],tot=o+n;
      return(<div key={i} className="flex items-center" style={{gap:8}}>
        <div style={{width:54,textAlign:"right",fontSize:10.5,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>{lab}</div>
        <div style={{flex:1,height:15,background:"#f2f1ec",borderRadius:3,overflow:"hidden",display:"flex"}}>
          <div style={{width:`${o/maxV*100}%`,background:ageColor(i),transition:"width .25s"}}/>
          <div style={{width:`${n/maxV*100}%`,background:MOSS,opacity:.55,transition:"width .25s"}}/></div>
        <div style={{width:48,fontSize:10.5,color:tot>0?SAGE:LINE_GRID,fontFamily:"var(--mono)",textAlign:"right"}}>{fmt(tot)}</div></div>);})}
    <div className="flex items-center" style={{gap:8,marginBottom:4}}>
      <div style={{width:54,textAlign:"right",fontSize:10,color:MOSS_DIM,letterSpacing:1}}>ALTER</div>
      <div style={{flex:1,fontSize:10,color:MOSS_DIM,letterSpacing:1}}>ANZAHL BÄUME →</div><div style={{width:48}}/></div></div>);
}

function SpeciesPanel({species,year}){
  const [speciesMode,setSpeciesMode]=useState("top_current");
  const [speciesLimit,setSpeciesLimit]=useState(20);
  const [manual,setManual]=useState(["Tilia europaea","Acer campestre","Quercus robur","Carpinus betulus","Acer platanoides"]);

  const rows=useMemo(()=>{
    const totalStart=species.rows.reduce((a,b)=>a+b.start,0)||1;
    const totalSel=species.rows.reduce((a,b)=>a+b.selected,0)||1;
    return species.rows.map(r=>{
      const startShare=r.start/totalStart*100;
      const selectedShare=r.selected/totalSel*100;
      const deltaAbs=r.selected-r.start;
      const deltaShare=selectedShare-startShare;
      const relDelta=r.start>0?deltaAbs/r.start*100:0;
      return {...r,startShare,selectedShare,deltaAbs,deltaShare,relDelta,absShareChange:Math.abs(deltaShare),absCountChange:Math.abs(deltaAbs)};
    });
  },[species]);

  const displayRows=useMemo(()=>{
    let arr=[...rows];
    if(speciesMode==="top_start") arr.sort((a,b)=>b.start-a.start);
    else if(speciesMode==="positive") arr.sort((a,b)=>b.deltaShare-a.deltaShare);
    else if(speciesMode==="negative") arr.sort((a,b)=>a.deltaShare-b.deltaShare);
    else if(speciesMode==="biggest") arr.sort((a,b)=>b.absShareChange-a.absShareChange);
    else if(speciesMode==="manual"){
      arr=arr.filter(r=>manual.includes(r.name));
      if(arr.length===0) arr=[...rows].sort((a,b)=>b.selected-a.selected).slice(0,speciesLimit);
    }else arr.sort((a,b)=>b.selected-a.selected);
    return arr.slice(0,speciesLimit);
  },[rows,speciesMode,speciesLimit,manual]);

  const chart=displayRows.map(r=>({
    art:r.label,
    heute:Math.round(r.start),
    jahr:Math.round(r.selected),
    delta:Math.round(r.deltaAbs),
    deltaShare:Math.round(r.deltaShare*10)/10,
    relDelta:Math.round(r.relDelta),
    fit:Math.round(r.climateFit*100),
    avgAge:r.avgAge,
    life:r.life,
    fallen:r.fallen,
  }));
  const selectList=useMemo(()=>[...rows].sort((a,b)=>b.absShareChange-a.absShareChange).slice(0,40),[rows]);
  const modeLabel={top_current:"grösster Bestand im Zieljahr",top_start:"grösster Bestand 2026",positive:"stärkste Zunahme",negative:"stärkste Abnahme",biggest:"stärkste Veränderung",manual:"eigene Auswahl"}[speciesMode];
  const chartHeight=Math.max(420,chart.length*30+80);
  const toggleManual=name=>setManual(a=>a.includes(name)?a.filter(x=>x!==name):[...a,name]);

  return(<>
    <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"18px 14px 8px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",padding:"0 8px 8px",gap:10,flexWrap:"wrap"}}>
        <div>
          <div style={{fontSize:13,color:SAGE,fontWeight:600}}>Baumarten / Artengruppen · heute vs. {year}</div>
          <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}>Ansicht: {modeLabel} · {displayRows.length} Arten</div>
        </div>
        <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>deterministische Artenschicht · alle Katasterarten</div>
      </div>

      <div style={{display:"flex",gap:10,flexWrap:"wrap",padding:"4px 8px 12px",alignItems:"center"}}>
        {[
          ["top_current","Top Zieljahr"],
          ["top_start","Top 2026"],
          ["positive","+ stärkste"],
          ["negative","− stärkste"],
          ["biggest","± grösste"],
          ["manual","Auswahl"]
        ].map(([id,lab])=>{const a=speciesMode===id;return(<button key={id} onClick={()=>setSpeciesMode(id)} style={{fontSize:12,padding:"6px 10px",borderRadius:8,cursor:"pointer",border:`1px solid ${a?MOSS:LINE_GRID}`,background:a?"rgba(225,0,15,0.07)":"transparent",color:a?MOSS:SAGE,fontFamily:"var(--mono)"}}>{lab}</button>);})}
        <label style={{fontSize:12,color:SAGE,fontFamily:"var(--mono)",marginLeft:"auto"}}>Anzahl&nbsp;
          <select value={speciesLimit} onChange={e=>setSpeciesLimit(parseInt(e.target.value,10))} style={{border:`1px solid ${LINE_GRID}`,borderRadius:7,padding:"5px 8px",color:PAPER,background:"#fff"}}>
            {[10,15,20,25,30].map(n=><option key={n} value={n}>{n}</option>)}
          </select>
        </label>
      </div>

      {speciesMode==="manual"&&<div style={{margin:"0 8px 12px",padding:"10px 12px",border:`1px solid ${LINE_GRID}`,borderRadius:12,background:"#faf9f5"}}>
        <div style={{fontSize:12,color:SAGE,fontWeight:600,marginBottom:8}}>Baumarten auswählen · Liste der 40 stärksten Veränderungen</div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(210px,1fr))",gap:6}}>
          {selectList.map(r=><label key={r.name} style={{fontSize:11.5,color:PAPER,display:"flex",alignItems:"center",gap:6,cursor:"pointer"}}>
            <input type="checkbox" checked={manual.includes(r.name)} onChange={()=>toggleManual(r.name)} style={{accentColor:MOSS}}/>
            <span>{r.label}</span>
            <span style={{marginLeft:"auto",fontFamily:"var(--mono)",color:r.deltaShare>=0?GREEN:BRICK}}>{r.deltaShare>0?"+":""}{r.deltaShare.toFixed(1)} %-P</span>
          </label>)}
        </div>
      </div>}

      <ResponsiveContainer width="100%" height={chartHeight}>
        <ComposedChart data={chart} layout="vertical" margin={{top:8,right:28,bottom:4,left:142}}>
          <CartesianGrid stroke={LINE_GRID} strokeDasharray="2 4" horizontal={false}/>
          <XAxis type="number" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} stroke={LINE_GRID}/>
          <YAxis type="category" dataKey="art" width={140} tick={{fill:TEXT_MUTE,fontSize:11}} stroke={LINE_GRID}/>
          <Tooltip contentStyle={{background:"#fff",border:`1px solid ${LINE_GRID}`,borderRadius:9,fontFamily:"var(--mono)",fontSize:12}} formatter={(value,name,props)=>[fmt(value),name]} labelFormatter={l=>l}/>
          <Bar dataKey="heute" name="2026" fill={BARK} radius={[0,4,4,0]} isAnimationActive={false}/>
          <Bar dataKey="jahr" name={String(year)} fill={MOSS} radius={[0,4,4,0]} isAnimationActive={false}/>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
    <div className="flex" style={{gap:12,marginTop:14,flexWrap:"wrap"}}>
      {[["Top-5-Anteil",Math.round(species.top5Share*100)+" %",PAPER],["Klima-Fit Proxy",Math.round(species.climateFitShare*100)+" %",GREEN],["Diversität",species.shannon.toFixed(2),MOSS],["Datenstatus","Kataster 2026",GREEN]].map(([l,val,c])=>(
        <div key={l} style={{background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"12px 15px",flex:"1 1 0",minWidth:120}}>
          <div style={{fontSize:11,color:TEXT_MUTE}}>{l}</div>
          <div style={{fontFamily:"var(--display)",fontSize:24,fontWeight:600,color:c,fontVariantNumeric:"tabular-nums"}}>{val}</div>
        </div>))}
    </div>
    <div style={{marginTop:14,background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"14px 18px"}}>
      <div style={{display:"flex",justifyContent:"space-between",gap:10,flexWrap:"wrap",alignItems:"baseline",marginBottom:8}}>
        <div style={{fontSize:13,color:SAGE,fontWeight:600}}>Veränderung der ausgewählten Baumarten</div>
        <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>Sortierung: {modeLabel}</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1.2fr .5fr .5fr .55fr .55fr .45fr .45fr",gap:8,fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",borderBottom:`1px solid ${LINE_GRID}`,paddingBottom:6,marginBottom:6}}>
        <div>Artengruppe</div><div>2026</div><div>{year}</div><div>Δ Anzahl</div><div>Δ Anteil</div><div>Alter</div><div>Leben</div>
      </div>
      {displayRows.map(r=>(<div key={r.name} style={{display:"grid",gridTemplateColumns:"1.2fr .5fr .5fr .55fr .55fr .45fr .45fr",gap:8,fontSize:12,padding:"5px 0",borderBottom:`1px solid ${LINE_GRID}55`}}>
        <div style={{color:PAPER}}>{r.label}</div>
        <div style={{fontFamily:"var(--mono)",color:SAGE}}>{fmt(r.start)}</div>
        <div style={{fontFamily:"var(--mono)",color:MOSS}}>{fmt(r.selected)}</div>
        <div style={{fontFamily:"var(--mono)",color:r.deltaAbs>=0?GREEN:BRICK}}>{r.deltaAbs>0?"+":""}{fmt(r.deltaAbs)}</div>
        <div style={{fontFamily:"var(--mono)",color:r.deltaShare>=0?GREEN:BRICK}}>{r.deltaShare>0?"+":""}{r.deltaShare.toFixed(1)} %-P</div>
        <div style={{fontFamily:"var(--mono)",color:SAGE}}>{r.avgAge} J</div>
        <div style={{fontFamily:"var(--mono)",color:SAGE}}>{r.life} J</div>
      </div>))}
      <div style={{fontSize:11,color:TEXT_MUTE,lineHeight:1.5,marginTop:10}}>{SPECIES_NOTE} Die Ansicht kann jetzt Top-Bestand, stärkste Zunahme, stärkste Abnahme, grösste Veränderung oder eine eigene Auswahl zeigen. Für finale Modelltreue sollten species_summary.csv/species_milestones.csv aus dem Python-Lauf verwendet werden.</div>
    </div>
  </>);
}


export default function App(){
  const exportRef = useRef(null);
  const [p,setP]=useState(DEFAULTS);
  const [view,setView]=useState("stock");
  const set=k=>v=>setP(s=>({...s,[k]:v}));
  const preset=n=>setP(s=>({...s,...PRESETS[n]}));
  const {series,det}=useMemo(()=>simulate(p),[p]);
  const yMax=useMemo(()=>Math.max(p.targetCount,...series.map(d=>d.p95))*1.06,[series,p.targetCount]);
  const ms=MILESTONES.map(off=>({off,pt:series[off]}));
  const t2030=series[4],v=verdict(t2030,p.targetCount),endPt=series[YEARS];
  const trendEnd=p.climateTrendEnd;
  const climateMult=(CLIMATE[p.scenario]?.[p.treeQ]?.[p.treeK])??1.0;
  const stress=climateMult*p.manualStress;
  const hToday=AVG_HAZARD_TODAY*stress*Math.pow(p.siteFactor,1)*Math.pow(p.mgmtFactor,0.6);
  const hEnd=hToday*(1+trendEnd);
  const deathsY1=Math.round(det.deaths[1]);
  const yOff=p.viewYearOff;
  const ageLine=useMemo(()=>det.medAge.map((mm,i)=>({year:CURRENT_YEAR+i,med:mm,young:Math.round(det.young[i]*100),old:Math.round(det.old60[i]*100)})),[det]);
  const renew=(()=>{const o=det.classOld[yOff].reduce((a,b)=>a+b,0),n=det.classNew[yOff].reduce((a,b)=>a+b,0);return n/(o+n||1);})();
  const species=useMemo(()=>simulateSpecies(p,det),[p,det]);

  async function exportPNG(){
    if(!exportRef.current) return;
    const canvas = await html2canvas(exportRef.current, {
      backgroundColor: PINE,
      scale: 2,
      useCORS: true,
      logging: false,
      windowWidth: exportRef.current.scrollWidth,
      windowHeight: exportRef.current.scrollHeight,
    });
    const link = document.createElement("a");
    link.download = `baumversorgung-dashboard-${view}-${CURRENT_YEAR}-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  function exportScenarioJSON(){
    const data = {
      exportedAt: new Date().toISOString(),
      view,
      parameters: p,
      milestones: MILESTONES.map(off => ({
        year: yearAt(off),
        p05: series[off].p05,
        p50: series[off].p50,
        p95: series[off].p95,
      })),
      currentVerdict: v,
      speciesDataset: SPECIES_DATASET,
      speciesTop: species.rows,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = `baumversorgung-szenario-${CURRENT_YEAR}-${Date.now()}.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  }

  function exportSeriesCSV(){
    const header = ["year","p05","p25","p50","p75","p95","mean"].join(";");
    const rows = series.map(d => [d.year,d.p05,d.p25,d.p50,d.p75,d.p95,d.mean].map(x => Math.round(x)).join(";"));
    const blob = new Blob([header + "\n" + rows.join("\n")], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = `baumversorgung-bestandsentwicklung-${CURRENT_YEAR}-${Date.now()}.csv`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  }

  return(<div ref={exportRef} style={{"--display":"'Inter',system-ui,sans-serif","--mono":"'JetBrains Mono',ui-monospace,monospace",
    background:PINE,minHeight:"100vh",color:PAPER,fontFamily:"'Inter',system-ui,sans-serif",padding:"clamp(16px,3vw,34px)"}}>
    <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
      input[type=range]{-webkit-appearance:none;appearance:none;background:${LINE_GRID};border-radius:99px;}
      input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:${MOSS};cursor:pointer;border:2px solid #fff;box-shadow:0 0 0 1px ${LINE_GRID};}
      input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:${MOSS};cursor:pointer;border:2px solid #fff;}`}</style>
    <div style={{maxWidth:1240,margin:"0 auto"}}>
      <div style={{marginBottom:18}}>
        <div className="flex items-center" style={{gap:12,marginBottom:8}}>
          <svg width="30" height="35" viewBox="0 0 30 35" aria-hidden="true">
            <path d="M1 1 H29 V22 C29 29 22 33 15 34 C8 33 1 29 1 22 Z" fill="#fff" stroke={PAPER} strokeWidth="1.4"/>
            {[8.5,19.5].map((cx,i)=>(<g key={i} fill={MOSS}>
              <ellipse cx={cx} cy="13" rx="3.1" ry="4"/>
              <circle cx={cx+(i?-2.6:2.6)} cy="9.2" r="1.7"/>
              <rect x={cx-1} y="16" width="2" height="7" rx="1"/>
              <path d={`M${cx+(i?-3:3)} 11 q${i?-3:3} 1 ${i?-2:2} 5`} stroke={MOSS} strokeWidth="1.4" fill="none"/>
            </g>))}
          </svg>
          <div style={{fontSize:11,letterSpacing:2.5,textTransform:"uppercase",color:MOSS_DIM,fontWeight:700}}>Stadt Winterthur · Stadtbaum-Modell v7</div>
        </div>
        <h1 style={{fontFamily:"var(--display)",fontSize:"clamp(28px,4.4vw,52px)",fontWeight:800,lineHeight:1.0,margin:"0 0 8px",letterSpacing:-1.2}}>
          {view==="stock"?"Wie entwickelt sich der Baumbestand?":view==="demography"?"Wie verschiebt sich die Altersstruktur?":"Wie verändert sich die Baumartenzusammensetzung?"}</h1>
        <div style={{width:48,height:3,background:MOSS,marginBottom:10}}/>
        <p style={{color:TEXT_MUTE,maxWidth:680,fontSize:14,lineHeight:1.5}}>
          {view==="stock"
            ?`Basis-Ausfall = kalibrierte Alters-Hazard aus dem Kataster (≈135 Fällungen/Jahr im Ist-Zustand). Ziel ${fmt(p.targetCount)} bis 2030.`
            :view==="demography"
              ?"Echte Altersverteilung aus den Pflanzjahren, fortgeschrieben mit der kalibrierten Alters-Hazard. Neue Bäume starten mit Alter "+p.initAge+"."
              :"Katasterbasierte Baumarten-/Artengruppen-Ansicht mit aktuellen Anteilen, mittlerem Alter, historischen Fällungen und Lebensdauer-Bandbreiten."}</p></div>

      <div className="flex" style={{gap:8,marginBottom:18,alignItems:"center",flexWrap:"wrap"}}>
        {[["stock","Bestand"],["demography","Demografie"],["species","Baumarten"]].map(([id,lab])=>(
          <button key={id} onClick={()=>setView(id)} style={{fontSize:13,padding:"9px 18px",borderRadius:9,cursor:"pointer",fontWeight:600,
            border:`1px solid ${view===id?MOSS:LINE_GRID}`,background:view===id?"rgba(225,0,15,0.07)":"transparent",color:view===id?MOSS:SAGE}}>{lab}</button>))}
        <div style={{flex:1}}/>
        <button onClick={exportPNG} style={{fontSize:12,padding:"9px 13px",borderRadius:9,cursor:"pointer",fontWeight:600,border:`1px solid ${MOSS}`,background:MOSS,color:"#fff"}}>PNG exportieren</button>
        <button onClick={exportSeriesCSV} style={{fontSize:12,padding:"9px 13px",borderRadius:9,cursor:"pointer",fontWeight:600,border:`1px solid ${LINE_GRID}`,background:"#fff",color:SAGE}}>CSV</button>
        <button onClick={exportScenarioJSON} style={{fontSize:12,padding:"9px 13px",borderRadius:9,cursor:"pointer",fontWeight:600,border:`1px solid ${LINE_GRID}`,background:"#fff",color:SAGE}}>Szenario</button>
      </div>

      {view==="stock"&&(
        <div style={{display:"flex",alignItems:"center",gap:16,flexWrap:"wrap",background:CARD,border:`1px solid ${v.color}55`,borderRadius:14,padding:"15px 20px",marginBottom:16}}>
          <div style={{width:12,height:12,borderRadius:99,background:v.color,boxShadow:`0 0 14px ${v.color}`}}/>
          <div style={{flex:1,minWidth:200}}><div style={{fontSize:12,color:TEXT_MUTE}}>Ziel 2030 · {fmt(p.targetCount)} Bäume</div>
            <div style={{fontFamily:"var(--display)",fontSize:23,fontWeight:600,color:v.color}}>{v.label}</div>
            <div style={{fontSize:12.5,color:TEXT_MUTE}}>{v.note}</div></div>
          <div style={{textAlign:"right",fontFamily:"var(--mono)"}}><div style={{fontSize:12,color:TEXT_MUTE}}>Median 2030</div>
            <div style={{fontSize:25,fontWeight:600,color:PAPER,fontVariantNumeric:"tabular-nums"}}>{fmt(t2030.p50)}</div></div></div>)}

      <div className="dash-grid" style={{display:"grid",gridTemplateColumns:"1fr",gap:18}}>
        <div>
          {view==="stock"&&(<>

            <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"18px 14px 8px"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",padding:"0 8px 8px"}}>
                <div style={{fontSize:13,color:SAGE,fontWeight:600}}>Bestandsentwicklung 2026 – 2126</div>
                <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>Median · 50 % · 90 % Band</div></div>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={series} margin={{top:8,right:14,bottom:4,left:6}}>
                  <CartesianGrid stroke={LINE_GRID} strokeDasharray="2 4" vertical={false}/>
                  <XAxis dataKey="year" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} ticks={[2026,2051,2076,2101,2126]} stroke={LINE_GRID}/>
                  <YAxis domain={[0,yMax]} tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} tickFormatter={x=>(x/1000).toFixed(0)+"k"} stroke={LINE_GRID} width={38}/>
                  <Tooltip content={<ChartTip target={p.targetCount}/>}/>
                  <Area dataKey="band90" stroke="none" fill={MOSS} fillOpacity={0.1} isAnimationActive={false}/>
                  <Area dataKey="band50" stroke="none" fill={MOSS} fillOpacity={0.22} isAnimationActive={false}/>
                  <Line dataKey="p50" stroke={MOSS} strokeWidth={2.4} dot={false} isAnimationActive={false}/>
                  <ReferenceLine y={p.targetCount} stroke={PAPER} strokeDasharray="5 4" strokeWidth={1.3} label={{value:`Ziel ${fmt(p.targetCount)}`,fill:PAPER,fontSize:11,position:"insideTopRight",fontFamily:"var(--mono)"}}/>
                  {MILESTONES.map(o=><ReferenceLine key={o} x={yearAt(o)} stroke={LINE_GRID} strokeDasharray="1 5"/>)}
                  <ReferenceDot x={2030} y={t2030.p50} r={4} fill={v.color} stroke="#fff" strokeWidth={2} isAnimationActive={false}/>
                </ComposedChart></ResponsiveContainer></div>
            <div className="flex" style={{gap:12,marginTop:14,flexWrap:"wrap"}}>{ms.map(({off,pt})=><MilestoneCard key={off} off={off} pt={pt} target={p.targetCount}/>)}</div>
            <div style={{marginTop:14,background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"14px 18px"}}>
              <div className="flex" style={{justifyContent:"space-between",alignItems:"flex-end",flexWrap:"wrap",gap:14}}>
                <div style={{flex:"1 1 240px"}}>
                  <div style={{fontSize:11,color:TEXT_MUTE,letterSpacing:1,textTransform:"uppercase",marginBottom:4}}>Kalibrierte Alters-Hazard (0–150 J)</div>
                  <HazardSpark/>
                  <div style={{fontSize:10.5,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}>jung ~0.3 % · reif ~1.5 % · Skala 0–4 %/J</div></div>
                <div style={{fontFamily:"var(--mono)",textAlign:"right"}}>
                  <div style={{fontSize:11,color:TEXT_MUTE}}>Ø Ausfall heute · {fmt(deathsY1)} Bäume/J</div>
                  <div><span style={{color:MOSS,fontSize:18,fontWeight:600}}>{(hToday*100).toFixed(2)} %</span>
                    <span style={{color:TEXT_MUTE}}> → </span><span style={{color:AMBER,fontSize:18,fontWeight:600}}>{(hEnd*100).toFixed(2)} %</span></div>
                  <div style={{fontSize:11,color:TEXT_MUTE,marginTop:6}}>Endbestand Median {fmt(endPt.p50)}</div></div></div></div>
          </>)}
          {view==="demography"&&(<>

            <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"18px 20px"}}>
              <div className="flex items-baseline" style={{justifyContent:"space-between",marginBottom:14,flexWrap:"wrap",gap:10}}>
                <div><div style={{fontSize:13,color:SAGE,fontWeight:600}}>Altersstruktur</div>
                  <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}><span style={{color:BARK}}>■</span> Bestand 2026 · <span style={{color:MOSS}}>■</span> seit 2026 gepflanzt</div></div>
                <div style={{textAlign:"right"}}><div style={{fontSize:11,color:TEXT_MUTE}}>angezeigtes Jahr</div>
                  <div style={{fontFamily:"var(--display)",fontSize:34,fontWeight:600,color:PAPER,lineHeight:1}}>{yearAt(yOff)}</div></div></div>
              <input type="range" min={0} max={YEARS} step={1} value={yOff} onChange={e=>set("viewYearOff")(parseInt(e.target.value))} style={{width:"100%",accentColor:MOSS,height:4,cursor:"pointer",marginBottom:18}}/>
              <Pyramid classOld={det.classOld[yOff]} classNew={det.classNew[yOff]}/></div>
            <div className="flex" style={{gap:12,marginTop:14,flexWrap:"wrap"}}>
              {[["Gesamtbestand",fmt(det.total[yOff]),PAPER],["Medianalter",det.medAge[yOff]+" J",MOSS],
                ["Anteil < 20 J",Math.round(det.young[yOff]*100)+" %",SAGE],["Anteil ≥ 60 J",Math.round(det.old60[yOff]*100)+" %",AMBER],
                ["seit 2026 gepflanzt",Math.round(renew*100)+" %",MOSS]].map(([l,val,c])=>(
                <div key={l} style={{background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"12px 15px",flex:"1 1 0",minWidth:110}}>
                  <div style={{fontSize:11,color:TEXT_MUTE}}>{l}</div>
                  <div style={{fontFamily:"var(--display)",fontSize:24,fontWeight:600,color:c,fontVariantNumeric:"tabular-nums"}}>{val}</div></div>))}</div>
            <div style={{marginTop:14,background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"16px 14px 8px"}}>
              <div style={{fontSize:13,color:SAGE,fontWeight:600,padding:"0 8px 8px"}}>Medianalter & Altersanteile über die Zeit</div>
              <ResponsiveContainer width="100%" height={230}>
                <ComposedChart data={ageLine} margin={{top:6,right:14,bottom:4,left:6}}>
                  <CartesianGrid stroke={LINE_GRID} strokeDasharray="2 4" vertical={false}/>
                  <XAxis dataKey="year" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} ticks={[2026,2051,2076,2101,2126]} stroke={LINE_GRID}/>
                  <YAxis yAxisId="l" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} stroke={LINE_GRID} width={32}/>
                  <YAxis yAxisId="r" orientation="right" domain={[0,100]} tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} tickFormatter={x=>x+"%"} stroke={LINE_GRID} width={40}/>
                  <Tooltip contentStyle={{background:"#ffffff",border:`1px solid ${LINE_GRID}`,borderRadius:9,fontFamily:"var(--mono)",fontSize:12}} labelStyle={{color:MOSS}}/>
                  <Line yAxisId="l" dataKey="med" name="Medianalter (J)" stroke={MOSS} strokeWidth={2.2} dot={false} isAnimationActive={false}/>
                  <Line yAxisId="r" dataKey="young" name="< 20 J (%)" stroke={GREEN} strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false}/>
                  <Line yAxisId="r" dataKey="old" name="≥ 60 J (%)" stroke={AMBER} strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false}/>
                  <ReferenceLine x={yearAt(yOff)} yAxisId="l" stroke={PAPER} strokeOpacity={.4}/>
                </ComposedChart></ResponsiveContainer></div>
          </>)}
          {view==="species"&&(<SpeciesPanel species={species} year={yearAt(yOff)}/>) }
        </div>

        <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"20px 20px 8px"}}>
          <div className="flex" style={{gap:7,flexWrap:"wrap",marginBottom:18}}>
            {Object.keys(PRESETS).map(n=>(<button key={n} onClick={()=>preset(n)} style={{fontSize:12,padding:"7px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${MOSS}40`,background:"rgba(225,0,15,0.05)",color:MOSS,fontWeight:600,textTransform:"capitalize"}}>{n}</button>))}
            <button onClick={()=>setP({...DEFAULTS,viewYearOff:p.viewYearOff})} style={{fontSize:12,padding:"7px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${LINE_GRID}`,background:"transparent",color:SAGE}}>Zurücksetzen</button></div>
          <Section title="Bestand & Ziel">
            <Slider label="Startbestand 2026" value={p.N0} min={12000} max={20000} step={100} onChange={set("N0")} display={fmt(p.N0)} hint="Kataster: 16'621 aktiv"/>
            <Slider label="Zielbestand 2030" value={p.targetCount} min={14000} max={20000} step={250} onChange={set("targetCount")} display={fmt(p.targetCount)}/>
          </Section>
          <Section title="Ausfall & Klima">
            <div style={{fontSize:11.5,color:TEXT_MUTE,marginBottom:12,lineHeight:1.5}}>Basis-Ausfall = kalibrierte Alters-Hazard (fix). Klima = TreeGOER-Exceedance gegen Winterthurs Zukunfts-bio05.</div>
            <Segmented label="Klimaszenario (CitiesGOER)" value={p.scenario}
              options={[{value:"ssp126",label:"SSP1-2.6"},{value:"ssp370",label:"SSP3-7.0"},{value:"ssp585",label:"SSP5-8.5"}]} onChange={set("scenario")}/>
            <Segmented label="TreeGOER-Quantil (tree_q)" value={p.treeQ}
              options={[{value:"q95",label:"q95 · milder"},{value:"qrt3",label:"qrt3 · strenger"}]} onChange={set("treeQ")}/>
            <Segmented label="TreeGOER-Stärke (tree_k)" value={p.treeK}
              options={[{value:0.08,label:"0.08"},{value:0.15,label:"0.15"}]} onChange={v=>set("treeK")(parseFloat(v))}/>
            <div style={{fontSize:11,color:SAGE,fontFamily:"var(--mono)",margin:"-4px 0 14px"}}>climate_mult = ×{climateMult.toFixed(3)} ({SCEN_LABEL[p.scenario]}, 45 % Artabdeckung)</div>
            <Slider label="Zusätzlicher Standort-/Stressfaktor" value={p.manualStress} min={0.7} max={2.0} step={0.05} onChange={set("manualStress")} display={"×"+p.manualStress.toFixed(2)} hint="für nicht erfasste lokale Stressoren; ×1.6 ≈ ~210 Fäll./J heute"/>
            <Slider label="Klima-Trend bis 2126" value={p.climateTrendEnd} min={0} max={1} step={0.05} onChange={set("climateTrendEnd")} display={"+"+(p.climateTrendEnd*100).toFixed(0)+" %"} hint="wirkt als ×(1+trend), linear über Zeit"/>
            <Slider label="Standortfaktor (^1.0)" value={p.siteFactor} min={0.7} max={1.5} step={0.05} onChange={set("siteFactor")} display={"×"+p.siteFactor.toFixed(2)}/>
            <Slider label="Managementfaktor (^0.6)" value={p.mgmtFactor} min={0.7} max={1.3} step={0.05} onChange={set("mgmtFactor")} display={"×"+p.mgmtFactor.toFixed(2)} hint="<1 = bessere Pflege"/>
          </Section>
          <Section title="Ersatz & Neupflanzung">
            <Slider label="Ersatzrate" value={p.replacementRate} min={0} max={1} step={0.05} onChange={set("replacementRate")} display={(p.replacementRate*100).toFixed(0)+" %"}/>
            <Slider label="Ersatzverzögerung" value={p.replacementDelay} min={0} max={5} step={1} onChange={set("replacementDelay")} display={p.replacementDelay+" J"}/>
            <Slider label="Zusatzpflanzungen / Jahr" value={p.annualNewTrees} min={0} max={800} step={25} onChange={set("annualNewTrees")} display={fmt(p.annualNewTrees)}/>
            <Slider label="Pflanzfenster von" value={p.newStart} min={1} max={10} step={1} onChange={set("newStart")} display={yearAt(p.newStart).toString()}/>
            <Slider label="Pflanzfenster bis" value={p.newEnd} min={1} max={30} step={1} onChange={set("newEnd")} display={yearAt(p.newEnd).toString()}/>
            <Slider label="Pflanzalter neue Bäume" value={p.initAge} min={1} max={20} step={1} onChange={set("initAge")} display={p.initAge+" J"} hint="new_tree_initial_age (Default 10)"/>
            <Segmented label="Pflanzstrategie" value={p.strategy} options={Object.keys(STRAT_NEW).map(k=>({value:k,label:STRAT_LABEL[k]}))} onChange={set("strategy")}/>
          </Section>
          <Section title="Lebensdauer & Simulation">
            <div style={{fontSize:11.5,color:TEXT_MUTE,marginBottom:12,lineHeight:1.5}}>life_mult = (130/Lebensdauer)^Gewicht, gekappt 0.5–2.0. Bestandsmittel ≈ 1.0 (Hauptwirkung: Unsicherheit).</div>
            <Slider label="Lebensdauer-Gewicht" value={p.lifeHazardWeight} min={0} max={1} step={0.1} onChange={set("lifeHazardWeight")} display={p.lifeHazardWeight.toFixed(1)}/>
            <Segmented label="Unsicherheitsmodus" value={p.lifeMode} options={[{value:"none",label:"none"},{value:"per_run",label:"per_run"}]} onChange={set("lifeMode")}/>
            <Slider label="Monte-Carlo-Läufe" value={p.nRuns} min={50} max={400} step={50} onChange={set("nRuns")} display={p.nRuns.toString()}/>
            <button onClick={()=>set("seed")(Math.floor(Math.random()*1e9))} style={{fontSize:12,padding:"7px 14px",borderRadius:8,cursor:"pointer",width:"100%",border:`1px solid ${LINE_GRID}`,background:"transparent",color:SAGE,marginTop:2}}>Zufall neu würfeln</button>
          </Section>
        </div>
      </div>

      <p style={{color:TEXT_MUTE,fontSize:12,lineHeight:1.6,marginTop:22,maxWidth:920}}>
        Repliziert die Engine aus winterthur_tree_stochastic_goal_planning_v7.py: p_fail = clip(base_p(Alter) · climate_mult · (1+trend) · Standort^1 · Management^0.6 · life_mult, 0.0001, 0.50).
        base_p = kalibrierte Alters-Hazard aus dem Kataster (Sterbetafel, Laplace α=0.5, min_risk_set=20, clip ≤ 0.35). climate_mult = bestandsgewichtete TreeGOER-Exceedance:
        exp(−tree_k·max(0, Winterthurs Zukunfts-bio05 − bio05-Grenze der Art)), 1/clip(.,0.2,2.0) — aus CitiesGOER (SSP126: 26.9 °C) und TreeGOER (45 % Artabdeckung, Rest neutral).
        Unter SSP126 ist climate_mult ≈ 1.0, der Klimaeffekt kommt also fast ganz aus dem climate_trend-Ramp. Neue Bäume starten mit Alter {p.initAge}, Ersatz = round(Ausfälle·Rate) nach Verzögerung.
        Bänder aus {p.nRuns} Läufen. Noch aggregiert statt artweise: base_p ist global (nicht je Art) und der CityTrees-Proxy wird durch TreeGOER überschrieben — wie in deiner Config.
      </p>
    </div>
    <style>{`@media(min-width:1000px){.dash-grid{grid-template-columns:1.55fr 1fr!important;}}`}</style>
  </div>);
}
