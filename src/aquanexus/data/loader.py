"""Load Japanese public-waters water quality monitoring data (公共用水域 検体値).

Source files come from the Ministry of the Environment's 水環境総合情報サイト or
from a prefecture publishing the same schema - Saitama's
``rNN_suishitsu_data.xlsx`` is the one this project uses. Each row is one grab
sample: 12 per station per year (see ``docs/DATA_SOURCES.md``).

Sheet layout
------------
The first seven rows are a header block, not data::

    row 0  category groups (一般項目, 生活環境項目, 健康項目 ...)
    row 1  parameter name, positioned at that parameter's 符合 column
    row 2  column names - direct fields, then repeating 符合 / 値 pairs
    row 3  units, at the 値 column
    row 4  significant figures
    row 5  decimal places
    row 6  detection limits (県定量下限値), at the 値 column
    row 7+ observations

Each measured parameter occupies two columns: 符合 (a qualifier, ``<`` below the
detection limit or ``>`` above the range) and 値 (the number). Columns are located
by matching header *text* rather than by fixed index, so the loader survives the
column shuffling that happens between publication years.

Censored values
---------------
Roughly 9,000 of the values in a year's file are flagged ``<``. The reported
number is then the detection limit, not a measurement. :func:`load_water_quality`
keeps the value as published and records the qualifier in a parallel
``<param>_flag`` column; :func:`apply_censoring` applies a substitution policy if
you want one. Silently treating a detection limit as a measurement biases every
downstream statistic upward, so the default is to make the choice explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.loader")

HEADER_ROWS = 7
_NAME_ROW, _COLUMN_ROW, _UNIT_ROW, _LIMIT_ROW = 1, 2, 3, 6

FLAG_COLUMN = "符合"
VALUE_COLUMN = "値"

# Direct (non-paired) fields, keyed by their header text in row 2.
DIRECT_FIELDS = {
    "測定機関": "agency",
    "水域名": "water_body",
    "地点名": "station",
    "水域": "water_body_code",
    "地点": "station_code",
    "西暦年": "year",
    "月日": "monthday",
    "時分": "hourmin",
    "調査区分": "survey_type",
    "採取位置": "sample_position",
    "天候": "weather",
    "流況": "flow_condition",
    "気温": "air_temp",
    "水温": "water_temp",
    "流量": "discharge",
    "採取水深": "sample_depth",
    "全水深": "total_depth",
}

# Parameters that feed habitat suitability. Anything not listed is still loaded,
# under its original Japanese name.
PARAMETER_NAMES = {
    "pH": "ph",
    "DO": "dissolved_oxygen",
    "BOD": "bod",
    "COD": "cod",
    "SS": "suspended_solids",
    "全窒素": "nitrogen_total",
    "全りん": "phosphorus_total",
    "大腸菌数": "coliform",
    "透視度（河川）\n透明度（湖沼）": "transparency",
    "底層DO": "bottom_do",
    "全亜鉛": "zinc_total",
}

# Categorical values arrive as "02晴れ" - a numeric code glued to its label.
_CODE_PREFIX = re.compile(r"^\d+")


@dataclass(frozen=True)
class Parameter:
    """One measured parameter and where to find it."""

    name: str
    column: str
    flag_index: int
    value_index: int
    unit: str = ""
    detection_limit: float = float("nan")


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _strip_code(value) -> str | float:
    """Turn '02晴れ' into '晴れ'. Leaves anything unprefixed alone."""
    if pd.isna(value):
        return np.nan
    return _CODE_PREFIX.sub("", str(value).strip()) or np.nan


def discover_parameters(header: pd.DataFrame) -> list[Parameter]:
    """Locate 符合/値 column pairs and name them from the row above."""
    names = header.iloc[_NAME_ROW]
    columns = header.iloc[_COLUMN_ROW]
    units = header.iloc[_UNIT_ROW]
    limits = header.iloc[_LIMIT_ROW]

    found: list[Parameter] = []
    for i in range(len(columns)):
        if _clean(columns.iloc[i]) != FLAG_COLUMN:
            continue
        if i + 1 >= len(columns) or _clean(columns.iloc[i + 1]) != VALUE_COLUMN:
            log.debug("flag column at %d has no matching value column", i)
            continue

        raw_name = _clean(names.iloc[i])
        if not raw_name:
            continue

        try:
            limit = float(limits.iloc[i + 1])
        except (TypeError, ValueError):
            limit = float("nan")

        found.append(
            Parameter(
                name=raw_name,
                column=PARAMETER_NAMES.get(raw_name, raw_name),
                flag_index=i,
                value_index=i + 1,
                unit=_clean(units.iloc[i + 1]),
                detection_limit=limit,
            )
        )
    return found


def _build_timestamp(frame: pd.DataFrame) -> pd.Series:
    """Combine 西暦年 + 月日 (MMDD) + 時分 (HHMM) into a timestamp.

    Sampling time is occasionally blank; those fall back to midnight rather than
    dropping the sample, since the date alone is enough for a monthly series.
    """
    year = pd.to_numeric(frame["year"], errors="coerce")
    monthday = pd.to_numeric(frame["monthday"], errors="coerce")
    hourmin = pd.to_numeric(frame["hourmin"], errors="coerce").fillna(0)

    valid = year.notna() & monthday.notna()
    stamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    parts = pd.DataFrame(
        {
            "year": year[valid].astype(int),
            "month": (monthday[valid].astype(int) // 100),
            "day": (monthday[valid].astype(int) % 100),
            "hour": (hourmin[valid].astype(int) // 100),
            "minute": (hourmin[valid].astype(int) % 100),
        }
    )
    stamp.loc[valid] = pd.to_datetime(parts, errors="coerce")
    return stamp


def load_water_quality(
    path: Path | str,
    sheet: int | str = 0,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """Load one 検体値 workbook into a tidy frame, one row per sample.

    Returns direct fields (station, timestamp, water temperature, discharge, ...)
    plus one column per parameter and a matching ``<param>_flag`` column carrying
    the censoring qualifier.

    ``parameters`` optionally restricts the loaded parameters, given either as
    canonical English names or the original Japanese.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if len(raw) <= HEADER_ROWS:
        raise ValueError(f"{path.name}: no observations below the header block")

    header, body = raw.iloc[:HEADER_ROWS], raw.iloc[HEADER_ROWS:].reset_index(drop=True)
    column_row = header.iloc[_COLUMN_ROW]

    out = pd.DataFrame(index=body.index)

    # Direct fields, located by header text.
    seen: set[str] = set()
    for i in range(len(column_row)):
        label = _clean(column_row.iloc[i])
        if label in DIRECT_FIELDS and label not in seen:
            out[DIRECT_FIELDS[label]] = body.iloc[:, i]
            seen.add(label)

    missing = set(DIRECT_FIELDS.values()) - set(out.columns)
    if missing:
        log.warning("%s: direct fields absent: %s", path.name, sorted(missing))

    out["timestamp"] = _build_timestamp(out)

    for field in ("weather", "flow_condition", "survey_type", "sample_position"):
        if field in out.columns:
            out[field] = out[field].map(_strip_code)

    for field in ("air_temp", "water_temp", "discharge", "sample_depth", "total_depth"):
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors="coerce")

    # Measured parameters.
    discovered = discover_parameters(header)
    if parameters is not None:
        wanted = set(parameters)
        discovered = [p for p in discovered if p.column in wanted or p.name in wanted]

    # Built as one block and concatenated once: a file carries ~100 parameters,
    # and inserting 200 columns individually fragments the frame badly.
    units: dict[str, str] = {}
    limits: dict[str, float] = {}
    measured: dict[str, pd.Series] = {}
    for param in discovered:
        measured[param.column] = pd.to_numeric(
            body.iloc[:, param.value_index], errors="coerce"
        )
        measured[f"{param.column}_flag"] = (
            body.iloc[:, param.flag_index].map(_clean).replace("", np.nan)
        )
        units[param.column] = param.unit
        limits[param.column] = param.detection_limit

    if measured:
        out = pd.concat([out, pd.DataFrame(measured, index=body.index)], axis=1)

    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    out.attrs["units"] = units
    out.attrs["detection_limits"] = limits
    out.attrs["source"] = path.name

    log.info(
        "%s: %d samples, %d parameters, %s to %s",
        path.name,
        len(out),
        len(discovered),
        out["timestamp"].min().date() if len(out) else "-",
        out["timestamp"].max().date() if len(out) else "-",
    )
    return out


def apply_censoring(
    frame: pd.DataFrame,
    policy: Literal["keep", "half", "zero", "nan"] = "half",
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a substitution policy to values flagged below the detection limit.

    ``half`` (the common convention) replaces them with half the reported limit,
    ``zero`` with 0, ``nan`` drops them, ``keep`` leaves them untouched. Values
    flagged ``>`` are never altered - an over-range reading is a real lower bound.
    """
    if policy == "keep":
        return frame

    out = frame.copy()
    targets = parameters or [
        c[:-5] for c in out.columns if c.endswith("_flag") and c[:-5] in out.columns
    ]

    for column in targets:
        flag = f"{column}_flag"
        if flag not in out.columns:
            continue
        below = out[flag].astype(str).str.startswith("<")
        if not below.any():
            continue
        if policy == "half":
            out.loc[below, column] = out.loc[below, column] / 2.0
        elif policy == "zero":
            out.loc[below, column] = 0.0
        else:
            out.loc[below, column] = np.nan
        log.debug("%s: %d censored values -> %s", column, int(below.sum()), policy)

    out.attrs = dict(frame.attrs)
    out.attrs["censoring"] = policy
    return out


def filter_stations(
    frame: pd.DataFrame,
    water_body: str | None = None,
    station: str | None = None,
) -> pd.DataFrame:
    """Substring filter on 水域名 / 地点名, e.g. ``water_body='綾瀬川'``."""
    out = frame
    if water_body:
        out = out[out["water_body"].astype(str).str.contains(water_body, na=False)]
    if station:
        out = out[out["station"].astype(str).str.contains(station, na=False)]
    return out.reset_index(drop=True)


def load_many(paths: list[Path | str], **kwargs) -> pd.DataFrame:
    """Load several yearly workbooks and concatenate them."""
    frames = [load_water_quality(p, **kwargs) for p in paths]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    merged_units: dict[str, str] = {}
    merged_limits: dict[str, float] = {}
    for f in frames:
        merged_units.update(f.attrs.get("units", {}))
        merged_limits.update(f.attrs.get("detection_limits", {}))
    combined.attrs["units"] = merged_units
    combined.attrs["detection_limits"] = merged_limits
    return combined.reset_index(drop=True)
