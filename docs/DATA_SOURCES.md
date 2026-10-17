# Data sources

Three sources, all open. Each result in this repository traces to files fetched from these.

## Dissolved oxygen: Ministry of the Environment public water body survey

- **What:** 公共用水域水質測定結果, river stations, 生活環境項目 (living environment items),
  FY2013-22, nationwide. Two file types: samples (`mk`, 検体値) and stations (`mm`, 水質測定点).
- **Where:** 水環境総合情報サイト (water-pub.env.go.jp), download page for measurement results.
- **How it was obtained:** by hand, through the site's web form, in a browser. The files are
  Shift-JIS (cp932) CSVs. The site names every downloaded zip randomly, so
  `scripts/unpack_moe.py <downloads_dir>` identifies each by its content and files it under
  `data/raw/moe/{mk,mm}_<fiscal year>/`.
- **Automated acquisition is not used.** The Ministry's river data portal (`river.go.jp`)
  explicitly prohibits it, and its backend accepts raw SQL fragments, so it is never called
  programmatically by this project. Do not add a scraper.
- **Result after loading** (`aquanexus.data.moe.build_dataset`): 565,674 DO samples, 5,883 stations,
  47 prefectures, every station with coordinates.

### Quirks that are easy to get wrong

- `NENDO` is the fiscal year; `FISCALYEAR` is the **calendar year of the sample**. A March 2023
  sample in FY2022 has `FISCALYEAR=2023`.
- `DAYTIME` is an unpadded `ddhhmm` integer (`30930` is the 3rd at 09:30).
- `DO_` is the published string (`<0.5` when below detection). `DO_2` zeroes censored values, so
  the loader parses `DO_`, halves a censored reading and flags it.
- `DO_3` is an exceedance flag derived from the environmental standard, which is derived from the
  label. It is kept as `below_standard` and **must never be a model input**.
- No water temperature and no river flow are in the files. Only 2.2% of samples are below 5 mg/L.

## Weather: NASA POWER

- **What:** daily 2 m air temperature, corrected precipitation, all-sky surface shortwave
  radiation and 10 m wind, from MERRA-2 and GPM IMERG, via the POWER point API.
- **Resolution:** values are fetched per 0.5° latitude by 0.625° longitude grid cell, 193 cells
  cover all stations, cached in `data/cache/weather/`. `scripts/fetch_weather.py` is resumable.
- **No leakage:** antecedent rain and temperature windows end the day before the sample.
- **Tried and dropped:** Open-Meteo ERA5 (the free tier rate-limits a ten-year request hard),
  ERA5-Land through its daily endpoint (temperature only).
- **Attribution:** data from the NASA Langley Research Center POWER Project, funded through the
  NASA Earth Science Division.

## Elevation: 国土地理院 (GSI)

- **What:** station elevation from GSI's national elevation model through its open elevation API,
  one request per station, four at a time. `scripts/fetch_elevation.py`, cached in
  `data/cache/elevation.csv`; 14 of 5,883 stations fall outside coverage and stay missing.
- **Derived:** relief among stations within 10 km and height relative to them
  (`aquanexus.data.geography`).
- Terms: 政府標準利用規約 (Japanese Government Standard Terms of Use).

## Land use: 国土数値情報 L03-b

- **What:** 土地利用細分メッシュ, FY2016 (平成28年), 100 m mesh, world geodetic datum, GML. One zip
  per 1st-order mesh (40 km by 80 km) from https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-b.html.
  The 113 files covering the stations are about 920 MB (`data/raw/landuse/`, gitignored).
- **How:** `scripts/fetch_landuse.py` reads the site's own listing page and downloads those files
  sequentially with a pause, resumable. The terms of use require a source credit and say nothing
  against scripted download; the files were fetched with the user's approval.
- **Derived:** for each station, the share of paddy, other farm, forest, barren, urban (buildings,
  roads, rail), other and water cells within 1 km and 5 km, over land cells only, plus the sea
  fraction and land cell count (`aquanexus.data.landuse`, `scripts/build_landuse.py`). These are
  circular buffers around the station, not upstream catchments: no flow-direction data is used.
- **Newest year offered in this layout is FY2016**, so land use is two to six years older than the
  samples; land use changes slowly, but this is a source of error.
- **Cite as:** 出典 国土交通省 国土数値情報（土地利用細分メッシュ）を加工して作成.

## What was considered and not used

- Distance to coast: would need coastline data.
- Fish and benthic surveys, river point clouds, HEC-RAS hydraulics: belonged to the earlier
  two-river version; see `docs/archive/`.
