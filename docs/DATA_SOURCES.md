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

## What was considered and not used

- Land use and catchment data (国土数値情報) and distance to coast: would need further downloads.
- Fish and benthic surveys, river point clouds, HEC-RAS hydraulics: belonged to the earlier
  two-river version; see `docs/archive/`.
