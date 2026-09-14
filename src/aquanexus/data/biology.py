"""Load 河川水辺の国勢調査 biological survey data.

Tokyo Metropolitan Government publishes its rounds of the National Census on
River Environments as species-by-site matrices: taxonomy down the rows, one
column per (survey year, river, site), and a mark where a taxon was recorded.

**Presence/absence, not abundance.** The usable response variable is therefore
taxon richness - how many taxa were recorded at a site in a survey year.

Sample size, stated plainly
---------------------------
This data cannot train a model and is not used as one. For the Ayase there are
**five fish records and six benthic records**, all but one at a single site
(内匠橋付近), across surveys spanning 1995-2024. Across all nine rivers in the
Tone file there are 46 fish and 36 benthic site-years, and the other eight rivers
have no hydraulic model.

Its value is as an **independent ecological check** on the synthetic habitat
labels: 内匠橋 is also one of the water quality stations this project already
loads, so observed richness there can be compared against HSI computed from
observed chemistry. Any such comparison must carry its n.

Source: 東京都建設局 河川水辺の国勢調査. File layout as published:

    row 0   survey year   (令和6年度 etc, sparse - forward fill)
    row 1   river name    (sparse - forward fill)
    row 2   site name
    row 3+  taxonomy (門/綱/目/科/種 和名) then one column per site-year
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.biology")

#: Japanese taxonomy headers, in the order they appear, mapped to English.
#: The published files do **not** all use the same depth: the benthic workbook
#: carries five ranks (門/綱/目/科/種) while the fish workbook carries three
#: (目/科/種). Assuming a fixed width silently reads the first two data columns
#: as taxonomy and misaligns every site - fish richness then reads as 1 taxon
#: per site-year instead of 6-20. The width is detected per file instead.
TAXONOMY_HEADERS = {
    "門和名": "phylum", "綱和名": "class", "目和名": "order",
    "科和名": "family", "種和名": "species",
}
TAXONOMY_COLUMNS = list(TAXONOMY_HEADERS.values())


def detect_taxonomy_columns(header_row: list) -> list[str]:
    """Return the taxonomy column names for this file, in order.

    Reads the leading cells of row 0 while they are recognised rank headers; the
    first non-rank cell is the start of the site-year data.
    """
    names = []
    for value in header_row:
        text = str(value).strip()
        if text in TAXONOMY_HEADERS:
            names.append(TAXONOMY_HEADERS[text])
        else:
            break
    if not names:
        raise ValueError("no taxonomy rank headers found in the first row")
    return names

#: Japanese era year -> Gregorian fiscal year.
_ERAS = {"令和": 2018, "平成": 1988, "昭和": 1925}


def parse_era_year(text: str) -> int | None:
    """Convert 令和6年度 or 平成15年度 to a Gregorian fiscal year.

    元年 (first year of an era) is written instead of 1, so it is handled
    explicitly rather than falling through the digit match.
    """
    if not text:
        return None
    match = re.match(r"(令和|平成|昭和)\s*(元|\d+)\s*年度?", str(text).strip())
    if not match:
        return None
    era, number = match.group(1), match.group(2)
    year = 1 if number == "元" else int(number)
    return _ERAS[era] + year


def _forward_fill(values: list) -> list:
    """Carry merged-cell headers across the columns they span."""
    out, current = [], None
    for value in values:
        text = str(value).strip()
        if text and text != "nan":
            current = text
        out.append(current)
    return out


def load_survey(path: Path | str, survey_type: str = "") -> pd.DataFrame:
    """Load one survey workbook into long form.

    Returns one row per (survey year, river, site, taxon) that was recorded,
    with the taxonomy columns alongside.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, header=None)
    if len(raw) < 4:
        raise ValueError(f"{path.name}: no data rows below the header block")

    taxonomy_columns = detect_taxonomy_columns(raw.iloc[0].tolist())
    n_taxonomy = len(taxonomy_columns)
    log.debug("%s: %d taxonomy rank(s): %s", path.name, n_taxonomy, taxonomy_columns)

    years = _forward_fill(raw.iloc[0].tolist())
    rivers = _forward_fill(raw.iloc[1].tolist())
    sites = raw.iloc[2].tolist()
    body = raw.iloc[3:].reset_index(drop=True)

    records = []
    for column in range(len(sites)):
        site = str(sites[column]).strip()
        if not site or site == "nan" or column < n_taxonomy:
            continue
        present = body.iloc[:, column].notna()
        if not present.any():
            continue
        for row in body.index[present]:
            taxonomy = {
                name: (str(body.iat[row, i]).strip()
                       if str(body.iat[row, i]) != "nan" else None)
                for i, name in enumerate(taxonomy_columns)
            }
            records.append({
                "survey_year": parse_era_year(years[column]),
                "survey_year_ja": years[column],
                "river": rivers[column],
                "site": site,
                "survey_type": survey_type or path.stem,
                **taxonomy,
            })

    frame = pd.DataFrame(records)
    log.info("%s: %d record(s), %d site-year(s), %d river(s)",
             path.name, len(frame),
             frame.groupby(["survey_year", "river", "site"]).ngroups if len(frame) else 0,
             frame["river"].nunique() if len(frame) else 0)
    return frame


def richness(frame: pd.DataFrame, level: str = "species") -> pd.DataFrame:
    """Taxon richness per (survey year, river, site).

    Counts distinct non-null taxa at ``level``. Rows whose entry at that level is
    a higher-rank placeholder (the files record e.g. a family name in the species
    column when identification stopped there) still count as one taxon, which is
    the convention these surveys use.
    """
    if frame.empty:
        return pd.DataFrame(columns=["survey_year", "river", "site", "richness"])

    grouped = (
        frame.dropna(subset=[level])
        .groupby(["survey_year", "survey_year_ja", "river", "site", "survey_type"])[level]
        .nunique()
        .reset_index(name="richness")
    )
    return grouped.sort_values(["river", "site", "survey_year"]).reset_index(drop=True)


def load_all(directory: Path | str) -> pd.DataFrame:
    """Load every survey workbook in a directory and concatenate."""
    directory = Path(directory)
    frames = []
    for path in sorted(directory.glob("*.xls*")):
        kind = "fish" if path.stem.startswith("gyo") else (
            "benthic" if path.stem.startswith("teisei") else path.stem
        )
        try:
            frames.append(load_survey(path, survey_type=kind))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            log.error("%s failed: %s", path.name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# The national census file - abundance rather than presence
# ---------------------------------------------------------------------------
#
# Everything above reads the Tokyo Metropolitan Government's species-by-site
# matrices, which mark a taxon as present or absent. MLIT publishes the same
# census nationally as one workbook per region, and that version carries the
# **count** and the temperature, velocity and depth measured at the moment of
# the survey. `scripts/fish_survey.py` pulls the Ayase rows out of the Kanto
# file; this reads what it wrote.
#
# It is more data, not enough data: five survey years between 1998 and 2019 at
# four sites. See docs/BIOLOGICAL_DATA.md before drawing anything from it.

NATIONAL_COLUMNS = (
    "survey_year", "river", "site", "km_from", "km_to", "species_ja", "count",
    "water_temp", "velocity_cm_s", "depth_cm",
)


def load_national_fish(path: Path | str) -> pd.DataFrame:
    """Read the extracted national-census fish records.

    Raises rather than returning something plausible when a column is missing:
    the file is generated by a script against a published workbook, and a
    silently absent `count` would turn abundance into presence without saying so.
    """
    frame = pd.read_csv(path)
    missing = [column for column in NATIONAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing {missing}; "
                         "re-run scripts/fish_survey.py")

    for column in ("count", "water_temp", "velocity_cm_s", "depth_cm",
                   "km_from", "km_to"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    log.info("national census: %d record(s), %d species, %d site(s), years %s",
             len(frame), frame["species_ja"].nunique(), frame["site"].nunique(),
             sorted(frame["survey_year"].unique()))
    return frame


def abundance(frame: pd.DataFrame) -> pd.DataFrame:
    """Individuals and species counted per (site, survey year).

    Counts are **effort-dependent**: the census uses whatever combination of
    methods suits the site - cast net, seine, electrofishing - and records the
    gear rather than a standardised unit of effort. Two sites in one year are
    therefore comparable in composition and only loosely in magnitude, so both
    numbers are carried and neither is called density.
    """
    if frame.empty:
        return pd.DataFrame(columns=["site", "survey_year", "individuals", "species"])

    return (
        frame.groupby(["site", "survey_year"])
        .agg(individuals=("count", "sum"), species=("species_ja", "nunique"))
        .reset_index()
        .sort_values(["site", "survey_year"])
        .reset_index(drop=True)
    )
