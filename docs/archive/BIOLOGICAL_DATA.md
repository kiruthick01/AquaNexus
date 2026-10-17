# Measured biology for the Ayase — what exists, and what it can settle

**Date verified:** 2026-09-14
**Method:** direct download of the published files; row counts computed from them, not quoted.
**Status:** an audit, not a result. Nothing in the model has changed because of it.

The habitat index this project serves is trained on labels the project generated
from response curves it wrote. That is stated everywhere the index appears, and
the only thing constraining those labels so far has been a falsification test and
a presence/absence extract with **n = 6 for the Ayase** (`src/aquanexus/data/biology.py`,
Tokyo Metropolitan Government files).

There is more, and it is better. This document says exactly how much better, and
where it stops.

---

## 1. The national river census — fish, with counts

**Source:** 河川水辺の国勢調査 (National Census on River Environments), MLIT,
published by NILIM.
**Entry point:** https://www.nilim.go.jp/lab/fbg/ksnkankyo/dl_00_index.html
**File used:** `download/slist/RT83_B01.zip` — Kanto region, river version, fish.
11.1 MB, one `.xlsx`, 53,736 rows across 44 rivers. No registration.
**Puller:** `scripts/fish_survey.py` → `data/processed/ayase_fish_survey.csv`.

> The census search UI at `www5.river.go.jp` does not resolve from this machine;
> the regional files linked above do, and they are the same data in bulk form.

### What it holds for the Ayase

**286 records, 39 species, 4 sites, 5 survey years.** Every record carries an
individual count and the water temperature, velocity and depth measured at the
survey, plus season, date, gear, channel form and river kilometre.

| Survey year | Records | Species | Individuals |
|---|---|---|---|
| 1998 | 32 | 12 | 276 |
| 2003 | 68 | 23 | 641 |
| 2009 | 60 | 19 | 241 |
| 2014 | 61 | 16 | 593 |
| 2019 | 65 | 16 | 287 |

| Site | River km | Survey years |
|---|---|---|
| 内匠橋 | 8.0–9.0 | 1998, 2003, 2009, 2014, 2019 |
| 八条大橋 | 13.8–14.3 | 2003 |
| 古綾瀬川合流点 | 13.8 | 1998 |
| 松原大橋 | 14.7–15.5 | 1998, 2003, 2009, 2014, 2019 |

Measured alongside: water temperature 17.3–28.5 °C, velocity −10 to 99 cm/s, depth
30–999 cm. **Fifteen records carry a negative velocity** — flow running upstream,
which is what a tidal reach does and which §3 returns to.

This is a strict improvement on what the repository had: **abundance instead of
presence, four sites instead of one, and hydraulics measured in the same hour as
the catch** rather than modelled for the same month.

---

## 2. What it can settle, and what it cannot

**It can test the profile assumption.** The synthetic index is defined for a
"lowland warmwater cyprinid assemblage" (`aquanexus.data.synthetic.LOWLAND_WARMWATER`).
The ten most abundant species recorded are:

| Species | Individuals |
|---|---|
| マハゼ (yellowfin goby) | 569 |
| ボラ (flathead grey mullet) | 297 |
| ヌマチチブ (dark sleeper) | 249 |
| モツゴ (topmouth gudgeon) | 195 |
| カダヤシ (mosquitofish) | 189 |
| ウキゴリ (floating goby) | 186 |
| タイリクバラタナゴ (rosy bitterling) | 39 |
| アベハゼ | 38 |
| ギンブナ (silver crucian carp) | 35 |
| アシシロハゼ | 34 |

Estuarine and diadromous species outnumber the rest at three of the four sites —
479 to 84 at 内匠橋, 693 to 317 at 松原大橋. The cyprinids the index is written
for (モツゴ, ギンブナ, タイリクバラタナゴ, コイ) are present and are not the
assemblage.

**Every surveyed site is marked 感潮あり — tidal.** That is a flag for the
hydraulic model as much as for the index: `aquanexus.hecras.project` writes a
**normal-depth downstream boundary**, which is correct for a reach without a
controlling downstream water level and wrong for one under tidal control. Whether
it matters here depends on §3.

**It cannot be a training label.** Five survey years over twenty-one, four sites,
counts that depend on gear and on effort the census records as text rather than as
a standardised unit. A regressor fitted on that would be fitting the survey
programme.

**It cannot validate a monthly prediction.** The model answers for a month; the
census answers for a morning in one season of one year.

---

## 3. The open question, and how to close it

**Do the surveyed sites lie inside the modelled reach?**

The census gives river kilometres (8.0–15.5 km); the HEC-RAS model indexes
sections as metres along its own centreline, 0 at the downstream end of the
modelled 27.5 km. The two are not known to share a datum, so the overlap is
**unverified** — and everything above hangs on it:

- If they overlap, then part of the modelled reach is tidal, the normal-depth
  boundary is questionable there, and the measured assemblage is a direct check
  on the index over the same water.
- If the survey sites are downstream of the modelled reach, the census describes
  a different river from the one being predicted, and its value drops to context.

One piece of evidence points at overlap and is not proof: **内匠橋 is both a
census site and one of the four water quality stations this project already
trains on** (`52内匠橋`, 48 of the 138 observations). Same name, same river; that
is suggestive, not a coordinate.

**How to settle it:** the same download page publishes the GIS version,
`download/shape/RG83_B01.zip` — survey sites as shapefiles. Reproject to
EPSG:6677 (the project CRS), snap each site to the modelled centreline, and read
off the station. That is an afternoon's work and it needs the `geo` extra, which
is deliberately not installed in CI.

---

## 4. Low-flow observations — the other missing measurement

Ranked second in the debt register, and worth recording what was checked on the
same day.

**The gap, exactly.** Of the 138 training observations, **17 sit below 2 m³/s and
every one of them is from a single station** (`55畷橋`); 9 are below 1 m³/s. The
model is biased −2.26 mg/L in that band, which is where oxygen stress matters.

**Saitama's per-sample workbook is the only open source that carries both
discharge and water temperature with each sample**, and the prefecture publishes
**three fiscal years on a rolling basis** — R04, R05, R06, which is exactly what
`scripts/download_data.py` already fetches. Older years exist only as PDF
summaries (annual means, 75th-percentile BOD), not per-sample rows. Verified by
probing the earlier filename pattern (`r01`/`r02`/`r03`/`h30` → 404) and by
reading the index page, which lists three files.

**The national archive goes back to 1984 and cannot help.** 環境省 水環境総合情報サイト
offers 検体値 (individual samples) for fiscal years **1984–2024**, 47 prefectures.
Its record layout (`download/kousui/kousui_k_manual.pdf`) is pH, DO, BOD, COD, SS,
coliforms, hexane extract, total zinc, the health items and TN/TP — **no 流量 and
no 水温**. This dataset joins chemistry to hydraulics through the discharge
measured beside the sample, and water temperature is the strongest single
correlate of the label it predicts (|r| = 0.68, second only to the saturation
concentration derived from it). A file with neither cannot enter it.

**So the low-flow band grows by about six samples a year, in March, when Saitama
publishes the next fiscal year** — `55畷橋` is sampled twelve times a year and
roughly half those samples fall below 2 m³/s — or by field measurement, or by pairing
national samples with an MLIT gauge, which is the gauged rating curve already
ranked first in the debt register and still a manual download from a site that
prohibits automated acquisition.

---

## 5. What would have to happen for the index to stop being synthetic

Stated plainly, because "we have fish data now" is the wrong conclusion to draw
from §1:

1. Settle §3 — do the surveys describe the modelled water?
2. Define a response the data can actually support. Richness or the
   estuarine/freshwater ratio per site-year is defensible; per-species abundance
   is not, at this effort.
3. Compare that response against the index computed from the observed chemistry
   at the nearest station and date, and report it with its n — 12 site-years, not
   286.
4. Expect it to disagree. The index assumes an assemblage the census does not
   record at these sites, and saying so with a number would be worth more than
   the index it corrects.

None of this is done. The habitat index remains SYNTHETIC everywhere it appears.
