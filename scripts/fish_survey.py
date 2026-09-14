"""Fetch the fish survey records for the Ayase from the national river census.

`河川水辺の国勢調査` (National Census on River Environments, MLIT) surveys fish,
benthic invertebrates, plants and birds on every Class A river system on a
roughly five-year cycle, and publishes the confirmed-species lists as one
workbook per region. The Kanto river-version fish file is a single 11 MB zip,
downloadable without registration, and it contains the Ayase.

That matters here because this project's habitat index is trained on labels it
generated itself. These records are the first **measured** biology in the
repository: per survey, per site, a species and a count, with the water
temperature, velocity and depth measured at the same time.

What it is not is a drop-in label. Read `docs/BIOLOGICAL_DATA.md` before using
it: five snapshots over 21 years, four sites, counts that depend on the capture
method and the effort behind it, and an assemblage at the downstream sites that
is not the one the synthetic index assumes.

    python scripts/fish_survey.py            # download, extract, summarise
    python scripts/fish_survey.py --summary  # summarise what is already here

Writes `data/processed/ayase_fish_survey.csv`.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile

import pandas as pd

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("scripts.fish_survey")

#: Kanto region, river version, fish. The code is <R|D><T|G><region><item>:
#: RT = river/species-list, 83 = Kanto, B01 = fish. B02 is benthic invertebrates,
#: which is the obvious next pull and is not used yet.
SOURCE_URL = (
    "https://www.nilim.go.jp/lab/fbg/ksnkankyo/download/slist/RT83_B01.zip"
)
USER_AGENT = "AquaNexus/0.1 (research; +https://github.com/kiruthick01/AquaNexus)"

#: Columns kept, renamed. The workbook carries ~90, most of them bank-treatment
#: flags that describe the survey site rather than the catch.
COLUMNS = {
    "調査年度": "survey_year",
    "河川名": "river",
    "地区名": "site",
    "距離_自(km)": "km_from",
    "距離_至(km)": "km_to",
    "種名": "species_ja",
    "個体数": "count",
    "季節": "season",
    "調査年月日(自)": "surveyed_on",
    "水温(℃)": "water_temp",
    "流速(cm/s)": "velocity_cm_s",
    "水深(cm)": "depth_cm",
    "PH": "ph",
    "透明度": "transparency",
    "魚類調査方法": "method",
    "捕獲方法": "capture_method",
    "感潮の有無": "tidal",
    "河川形態": "channel_form",
}

OUTPUT = "ayase_fish_survey.csv"


def download(url: str = SOURCE_URL) -> bytes:
    log.info("fetching %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    log.info("%.1f MB", len(payload) / 1e6)
    return payload


def extract(payload: bytes, river: str = "綾瀬") -> pd.DataFrame:
    """Pull one river's rows out of the regional workbook."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    frame = pd.read_excel(io.BytesIO(archive.read(name)), sheet_name="データ")
    log.info("%s: %d row(s) across %d river(s)", name, len(frame),
             frame["河川名"].nunique())

    rows = frame[frame["河川名"].astype(str).str.contains(river, na=False)]
    if rows.empty:
        raise SystemExit(f"no rows for {river} - the workbook layout may have changed")

    missing = [c for c in COLUMNS if c not in rows.columns]
    if missing:
        raise SystemExit(f"columns missing from the workbook: {missing}")

    return rows[list(COLUMNS)].rename(columns=COLUMNS).reset_index(drop=True)


def summarise(rows: pd.DataFrame) -> None:
    """Print what the records do and do not cover.

    Deliberately verbose about the second part. A count of 569 マハゼ reads as a
    dataset until you notice it is five mornings spread over twenty-one years.
    """
    log.info("%d record(s), %d species, %d site(s)",
             len(rows), rows["species_ja"].nunique(), rows["site"].nunique())

    by_year = rows.groupby("survey_year").agg(
        records=("species_ja", "size"),
        species=("species_ja", "nunique"),
        individuals=("count", "sum"),
    )
    log.info("by survey year:\n%s", by_year.to_string())

    sites = rows.groupby("site")[["km_from", "km_to"]].agg(["min", "max"])
    log.info("sites (river km):\n%s", sites.to_string())

    top = rows.groupby("species_ja")["count"].sum().sort_values(ascending=False)
    log.info("most abundant:\n%s", top.head(10).to_string())

    for column, unit in (("water_temp", "°C"), ("velocity_cm_s", "cm/s"),
                         ("depth_cm", "cm")):
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if len(values):
            log.info("%s: %.1f-%.1f %s over %d record(s)", column, values.min(),
                     values.max(), unit, len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true",
                        help="summarise the extracted file instead of downloading")
    args = parser.parse_args()

    settings.ensure_dirs()
    path = settings.PROCESSED_DIR / OUTPUT

    if args.summary:
        if not path.is_file():
            log.error("%s does not exist; run without --summary first", path)
            return 1
        summarise(pd.read_csv(path))
        return 0

    rows = extract(download())
    rows.to_csv(path, index=False, encoding="utf-8")
    log.info("wrote %s", path)
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
