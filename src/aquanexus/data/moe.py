"""Load the Ministry of the Environment's nationwide 公共用水域 sample files.

Two downloads from 水環境総合情報サイト (https://water-pub.env.go.jp/water-pub/mizu-site/mizu/download/download.asp)
are joined here:

``mk`` 検体値データ
    One row per grab sample. Keyed by ``ZETTAICODE`` (the station id), with the
    fiscal year, month and a ``DAYTIME`` field that packs day-of-month, hour and
    minute as an unpadded ``ddhhmm`` integer (``30930`` = the 3rd, 09:30).
``mm`` 水質測定点データ
    One row per station and year, with latitude and longitude.

Both files are cp932 CSVs inside a zip whose single member is ``output.csv``.

Value columns
-------------
Each parameter ``P`` comes as three columns. ``P`` is the published string and may
carry a ``<`` prefix when the reading was below the detection limit, ``P2`` is a
numeric copy in which those readings are zeroed, and ``P3`` is an exceedance flag
derived from the environmental standard. ``P3`` is computed from the label, so it
must never be used as a model input; it is kept only as ``below_standard``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.moe")

ENCODING = "cp932"

# Sample-file column -> output name. The first of each pair is the published string.
PARAMETERS = {
    "DO_": "dissolved_oxygen",
    "BOD": "bod",
    "COD": "cod",
    "SS": "ss",
    "PH": "ph",
}

DO_RANGE = (0.0, 25.0)


def _read(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=ENCODING, low_memory=False)


def _parse_censored(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a published value column into a number and a below-detection flag.

    A ``<x`` reading becomes ``x / 2`` (the usual half-detection-limit convention)
    and is flagged.
    """
    text = raw.astype("string").str.strip()
    censored = text.str.startswith("<").fillna(False).astype(bool)
    number = pd.to_numeric(text.str.lstrip("<"), errors="coerce")
    number = number.where(~censored, number / 2.0)
    return number.astype(float), censored


def _unpack_daytime(daytime: pd.Series) -> pd.DataFrame:
    value = pd.to_numeric(daytime, errors="coerce")
    return pd.DataFrame(
        {
            "day": (value // 10000).astype("Int64"),
            "hour": ((value // 100) % 100).astype("Int64"),
            "minute": (value % 100).astype("Int64"),
        }
    )


def read_samples(path: Path | str) -> pd.DataFrame:
    """Read one ``mk`` file (zip or csv) into a tidy sample table."""
    raw = _read(path)
    when = _unpack_daytime(raw["DAYTIME"])

    out = pd.DataFrame(
        {
            "station_id": raw["ZETTAICODE"].astype("int64"),
            "pref_code": raw["PREFCODE"].astype("int64"),
            # The file's names are misleading: NENDO is the fiscal year (April start)
            # and FISCALYEAR is the calendar year of the sample.
            "fiscal_year": raw["NENDO"].astype("int64"),
            "year": raw["FISCALYEAR"].astype("int64"),
            "month": raw["MONTH"].astype("int64"),
        }
    )
    out["day"] = when["day"]
    out["hour"] = when["hour"]
    out["depth_class"] = raw["DEPTH"]

    for column, name in PARAMETERS.items():
        value, censored = _parse_censored(raw[column])
        out[name] = value
        if name in {"dissolved_oxygen", "bod", "cod", "ss"}:
            out[f"{name}_censored"] = censored

    out["below_standard"] = raw["DO_3"].fillna(0).astype(int).astype(bool)

    stamp = pd.to_datetime(
        {"year": out["year"], "month": out["month"], "day": out["day"]},
        errors="coerce",
    )
    out["date"] = stamp
    return out


def read_stations(path: Path | str) -> pd.DataFrame:
    """Read one ``mm`` file into a station table (one row per station-year)."""
    raw = _read(path)
    return pd.DataFrame(
        {
            "station_id": raw["zettaicode"].astype("int64"),
            "fiscal_year": raw["nendo"].astype("int64"),
            "name": raw["locationname"],
            "water_body": raw["water"],
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
        }
    )


def load_samples(paths: list[Path | str]) -> pd.DataFrame:
    """Concatenate yearly ``mk`` files. Each row appears once per station-date-depth."""
    frames = [read_samples(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(["station_id", "date", "hour", "depth_class", "dissolved_oxygen"])
    log.info(
        "%d sample rows from %d files (%d duplicates dropped)",
        len(out),
        len(frames),
        before - len(out),
    )
    return out


def load_stations(paths: list[Path | str]) -> pd.DataFrame:
    """Concatenate yearly ``mm`` files and keep one row per station.

    A station can appear under several water bodies in one year, and its recorded
    position can shift slightly between years. The latest year with a coordinate
    wins, which is the position the agency currently considers correct.
    """
    frames = [read_stations(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["latitude", "longitude"])
    out = out.sort_values(["station_id", "fiscal_year"]).drop_duplicates("station_id", keep="last")
    return out.reset_index(drop=True)


def build_dataset(samples: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join samples to station coordinates and keep rows usable as DO labels.

    Dropped, with counts logged: no DO value, no valid date, no coordinate, and DO
    outside the physical range. Below-detection DO readings are kept and flagged -
    a river at the detection limit is a real, low-oxygen observation.
    """
    out = samples.merge(
        stations[["station_id", "name", "water_body", "latitude", "longitude"]],
        on="station_id",
        how="left",
    )
    n0 = len(out)
    steps = {
        "no DO": out["dissolved_oxygen"].notna(),
        "no date": out["date"].notna(),
        "no coordinate": out["latitude"].notna(),
        "DO out of range": out["dissolved_oxygen"].between(*DO_RANGE),
    }
    keep = pd.Series(True, index=out.index)
    for label, mask in steps.items():
        dropped = int((keep & ~mask).sum())
        if dropped:
            log.info("dropped %d rows: %s", dropped, label)
        keep &= mask
    out = out[keep].sort_values(["station_id", "date", "hour"]).reset_index(drop=True)
    log.info("%d of %d rows kept, %d stations", len(out), n0, out["station_id"].nunique())
    return out


def find_files(directory: Path | str, kind: str) -> list[Path]:
    """Locate extracted ``output.csv`` files for ``kind`` in ``('mk', 'mm')``."""
    return sorted(Path(directory).glob(f"{kind}_*/output.csv"))
