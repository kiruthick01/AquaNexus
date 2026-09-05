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

## Open questions before Phase 1

- [ ] Confirm 綾瀬川/中川 tile coverage is continuous enough for a contiguous reach
- [ ] Check whether 埼玉県 常時監視 stations publish sub-daily series (would materially help)
- [ ] Identify specific MLIT gauge(s) on 綾瀬川/中川 for discharge boundary conditions
- [ ] Decide reach length and number of cross-sections from actual tile extent
