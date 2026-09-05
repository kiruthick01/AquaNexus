# AquaNexus — Data Source Availability Audit

**Date verified:** 2026-09-06
**Method:** Direct HTTP requests to each source; CKAN API queries against G空間情報センター.
**Status:** Supersedes `ML_STRATEGY.md` §2.2–2.4, which was written from assumption rather than verification.

---

## Summary of changes to the plan

| Plan assumption | Reality |
|---|---|
| Yodo River, "closest to IGES" | Yodo is Osaka/Kyoto, ~400 km from IGES (Hayama, Kanagawa). No published channel geometry found. |
| NIES database, Yodo CSV download | Does not exist in that form. NIES public-waters download stops at **FY2009**. |
| MLIT river.go.jp bulk download | Site **explicitly prohibits automated acquisition**. Manual only, 30-day cap per request. |
| 35,040 hourly records from 4-year simulation | Not supportable: WQ observations are monthly, so hourly boundary conditions would be interpolation artifacts. |
| HEC-RAS water quality modules | ✅ Confirmed — v6 ships Temperature, GCSM, NSMI, NSMII. |

---

## 1. Channel geometry (the previous blocker) — SOLVED

### 埼玉県 河川点群データ (Saitama Prefecture River Point Cloud) ★ SELECTED
- **URL:** https://www.geospatial.jp/ckan/dataset/river-pointcloud-saitama
- **License:** **CC BY 4.0** (permissive — attribution only)
- **Acquisition:** UAV drone + **narrow multibeam echosounder**, integrated
- **Why this matters:** multibeam gives the *submerged bed*. GSI's 5m DEM has holes exactly at
  water surfaces because topographic LiDAR does not penetrate water. This dataset does not.
  It is the only open Japanese source found that supplies above- and below-water channel
  geometry as one product.
- **Period:** FY2022–2023 (令和4〜5年)
- **CRS:** JGD2011 / Japan Plane Rectangular CS Zone 9
- **Coverage:** 綾瀬川, 中川 (added 2024-06) + 毛長川, 辰井川 and 54 others (added 2025-08) — 58+ rivers
- **Format:** LAS/LAZ in ZIP, per map tile. **Average 520 MB/file, max 1.0 GB** — plan disk accordingly.
- **Browser:** https://experience.arcgis.com/experience/d88b12836c194e8dbaa73155a23d0400/

### Rejected geometry alternatives
| Source | Why rejected |
|---|---|
| GSI 5m DEM (DEM5A) | ~70% national coverage, but voids at rivers/water bodies. Unusable for channel. |
| 国土数値情報 W05 河川データ | Centerline polylines only. No cross-sections, no bathymetry. |
| MLIT LAS sets (天竜川, 狩野川水系, 菊川, 沼津, 静岡国道) | Ground-classified aerial LiDAR (`_grd`) — water surface, not bed. Also 独自利用規約, not CC. |

---

## 2. Water quality observations

### 環境省 水環境総合情報サイト ★ PRIMARY
- **URL:** https://water-pub.env.go.jp/water-pub/mizu-site/mizu/download/download.asp
- **Status:** ✅ Verified live and open. No access restriction.
- **Query axes:** fiscal year × prefecture × water body type (`1` = river) × parameter group × format (CSV/SHP/GML)
- **Record types:** `md` 年間値 (annual), **`mk` 検体値 (individual samples ← use this)**, `mm` 測定点 (station metadata)
- **Parameter groups:** 健康項目, 生活環境項目 (DO/BOD/pH/SS), 全燐・全窒素, トリハロメタン生成能, ダイオキシン
- **Frequency:** **Monthly.** ~12 samples/station/year. This is the binding constraint on the dataset design.
- **Note:** download is via the web form. The backend takes a raw SQL `where` fragment; do not
  construct requests against it programmatically.

### Supporting water quality sources
- 埼玉県 公共用水域水質常時監視 — https://www.pref.saitama.lg.jp/a0505/koukyouyousuiiki.html
  (常時監視 implies some automatic stations; worth checking for sub-daily temp/DO/pH/turbidity)
- 江戸川河川事務所 中川・綾瀬川 水質調査データ — https://www.ktr.mlit.go.jp/edogawa/edogawa00512.html
  (includes annual 流量 + 水質 surveys for mainstem and tributaries)
- GEMStat (NIES contribution) — https://gemstat.org/ — 20 stations nationwide, 2020–2023, via data request

---

## 3. Hydrology (discharge / stage)

### MLIT 水文水質データベース — CONSTRAINED
- **URL:** https://www1.river.go.jp/
- **Status:** ⚠️ Programmatic access returns:
  `Access Restrictions / This site prohibits data acquisition using tools, etc.`
- **Consequence:** **Do not scrape.** Manual UI download only, capped at 30 days per request.
- Third-party scraping tutorials exist; they operate against the stated terms.
- geospatial.jp dataset `9005` merely links back here — no bulk mirror exists.

### 水防災オープンデータ提供サービス — NOT FREE
- https://www.river.or.jp/koeki/opendata/index.html
- Telemetry + XRAIN radar, but requires application and **cost-sharing**. Not viable for a PoC.

---

## 4. Meteorological forcing

### 気象庁 過去の気象データ ★ CLEAN
- **URL:** https://www.data.jma.go.jp/risk/obsdl/
- **License:** 公共データ利用規約 第1.0版 — free reuse **with attribution**
- **Provides:** air temperature, solar radiation, precipitation — hourly
- Covers `solar_radiation` and `thermal_gradient` inputs in the §3.3 feature set.
- Caveat: JMA asks that high-frequency / bulk automated access be avoided.

---

## 5. Recommended target

**River:** 綾瀬川 (Ayase) and/or 中川 (Naka), Saitama — 利根川水系

Rationale:
1. Only rivers found with open, CC BY, **bathymetric** channel geometry.
2. Dedicated MLIT discharge + water quality survey programme (江戸川河川事務所).
3. Prefectural continuous monitoring on top of the national monthly programme.
4. **Strong ecological narrative:** 綾瀬川 held Japan's worst-BOD ranking for years and has
   measurably recovered under 清流ルネッサンス. A documented water-quality gradient over time is
   exactly the signal a habitat suitability model should be able to track — and it gives the
   project a real story rather than a synthetic one.
5. Kanto region — defensible proximity to IGES, unlike the Yodo.

**Dataset design:** designed experiment over flow space (per decision 2026-09-06), not a
4-year hourly timeline. HEC-RAS sweeps a grid of realistic flow/temperature conditions bounded
by observed ranges; real monthly observations anchor and validate those ranges.

---

## Open questions — resolved 2026-09-06

### ✅ Tile coverage is continuous. Ayase selected as primary.

Enumerated the full tile index by decoding the published vector tiles
(`https://gic-saitama.s3.ap-northeast-1.amazonaws.com/2025/Vectortile2026/river/{z}/{x}/{y}.pbf`),
which carry `MESH_NO` and a direct `URL` per tile. **1,385 tiles across 59 rivers.**

| | Ayase (綾瀬川) | Naka (中川) |
|---|---|---|
| Tiles | **89** | 53 |
| Chain length | **29.4 km** | 21.8 km |
| Median step between tiles | 296 m | 362 m |
| Max step | 866 m | 743 m |
| Gaps > 1 km | **0** | **0** |
| Longest unbroken run | 84 tiles | 53 (all) |
| Mean tile size | **107 MB** | 616 MB |
| Total download | **~9.5 GB** | ~32.7 GB |

Both are fully contiguous — no breaks. **Ayase is the primary reach**: 3.4× smaller download,
longer continuous extent, and the stronger ecological narrative. Naka is held as a
spatial-validation holdout.

Ayase tiles form two series: `ayasegawa-0240` … `ayasegawa-1050` in steps of 10 (82 tiles,
the contiguous mainstem) plus 7 tiles numbered `0011`–`0191` from a separate survey block.

**Reach decision:** ~29 km of Ayase mainstem. At 500 m cross-section spacing that yields
~58 sections — the original plan's "10 cross-sections" was arbitrary and is superseded.

### ✅ No sub-daily monitoring exists. Monthly confirmed — but richer than expected.

`常時監視` is the legal name of the monitoring *programme* under 水質汚濁防止法, **not**
high-frequency instrumentation. Verified against
[`r06_suishitsu_data.xlsx`](https://www.pref.saitama.lg.jp/documents/15286/r06_suishitsu_data.xlsx)
(FY2024 検体値, 1,257 rows × 221 columns): **exactly 12 samples per station per year.**

12 stations cover the system across 5 water bodies — 綾瀬川上流, 綾瀬川下流, 古綾瀬川,
中川上流, 中川中流.

**The useful surprise:** each sample row carries co-measured **水温 (water temp), 気温 (air
temp) and 流量 (discharge)** alongside the chemistry, at the same instant. So discharge–water
quality relationships can be built entirely from this one file, without joining to
`river.go.jp` at all. That materially reduces the manual-download burden — river.go.jp is
still needed for hourly discharge *boundary conditions*, but not for the observational anchor.

### Point cloud characteristics (verified against `ayasegawa-0610`)

- LAS 1.2, point format 2 (RGB present), ~2.6 M points per 270 × 356 m tile — **~27 pts/m²**
- **CRS: EPSG:6677 with `x` = easting, `y` = northing.** Verified by round-tripping the tile
  centroid: 10 m agreement, versus 16 km for the swapped (Japanese X=north) convention.
  Getting this backwards would misplace every cross-section.
- ⚠️ **All points are classification 1 (unclassified).** There is no ground class to filter
  on, so bed extraction must derive bare-earth itself (per-cell low quantile) rather than
  selecting `classification == 2`.
- Zip → LAS expands ~2×: a 33 MB zip yields a 68 MB LAS.

### Still open

- [ ] Identify MLIT gauge(s) on 綾瀬川 for hourly discharge boundary conditions (manual pull)
