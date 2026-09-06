"""Biological survey loader tests.

The headline case is taxonomy width. The published workbooks disagree - benthic
files carry five ranks, fish files three - and assuming a fixed width reads the
first two data columns as taxonomy, silently collapsing fish richness to 1 taxon
per site-year instead of the true 6-20. Nothing downstream would flag it.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.biology import (
    detect_taxonomy_columns,
    load_all,
    load_survey,
    parse_era_year,
    richness,
)

FISH_RANKS = ["目和名", "科和名", "種和名"]
BENTHIC_RANKS = ["門和名", "綱和名", "目和名", "科和名", "種和名"]


def build_workbook(path, ranks, columns, taxa):
    """columns: list of (year, river, site). taxa: list of (names..., present_cols)."""
    width = len(ranks) + len(columns)
    grid = [[np.nan] * width for _ in range(3)]

    for i, name in enumerate(ranks):
        grid[0][i] = name
    # Year and river headers are merged cells: written once, then blank.
    last_year = last_river = None
    for j, (year, river, site) in enumerate(columns):
        col = len(ranks) + j
        if year != last_year:
            grid[0][col] = year
            last_year = year
        if river != last_river:
            grid[1][col] = river
            last_river = river
        grid[2][col] = site

    body = []
    for names, present in taxa:
        row = [np.nan] * width
        for i, value in enumerate(names):
            row[i] = value
        for j in present:
            row[len(ranks) + j] = "○"
        body.append(row)

    pd.DataFrame(grid + body).to_excel(path, header=False, index=False)
    return path


# ---------------------------------------------------------------------------
# Era years
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("令和6年度", 2024), ("令和2年度", 2020), ("平成15年度", 2003),
    ("平成7年度", 1995), ("平成10年度", 1998), ("昭和60年度", 1985),
])
def test_era_years_convert(text, expected):
    assert parse_era_year(text) == expected


def test_gannen_is_year_one():
    """元年 is written instead of 1 and must not fall through the digit match."""
    assert parse_era_year("令和元年度") == 2019
    assert parse_era_year("平成元年度") == 1989


def test_unparseable_era_returns_none():
    assert parse_era_year("2024") is None
    assert parse_era_year("") is None
    assert parse_era_year(None) is None


# ---------------------------------------------------------------------------
# Taxonomy width
# ---------------------------------------------------------------------------


def test_detects_three_rank_fish_layout():
    assert detect_taxonomy_columns([*FISH_RANKS, "平成7年度", np.nan]) == [
        "order", "family", "species"
    ]


def test_detects_five_rank_benthic_layout():
    assert detect_taxonomy_columns([*BENTHIC_RANKS, "平成10年度"]) == [
        "phylum", "class", "order", "family", "species"
    ]


def test_no_rank_headers_raises():
    with pytest.raises(ValueError, match="no taxonomy rank headers"):
        detect_taxonomy_columns(["something", "else"])


def test_fish_richness_is_not_collapsed_by_assumed_width(tmp_path):
    """Regression: a fixed width of five read two data columns as taxonomy.

    Fish files have three ranks, so site 0 and site 1 were consumed as taxonomy
    and every remaining site reported a single taxon.
    """
    path = build_workbook(
        tmp_path / "gyo_test.xls", FISH_RANKS,
        [("平成7年度", "綾瀬川", "内匠橋付近"), ("平成7年度", "綾瀬川", "綾瀬水門")],
        [(["コイ目", "コイ科", f"種{i}"], [0] if i < 4 else [1]) for i in range(6)],
    )
    counts = richness(load_survey(path, "fish"))
    assert set(counts["richness"]) == {4, 2}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.fixture
def benthic(tmp_path):
    return build_workbook(
        tmp_path / "teisei_test.xls", BENTHIC_RANKS,
        [("平成10年度", "綾瀬川", "内匠橋付近"),
         ("令和2年度", "綾瀬川", "内匠橋付近"),
         ("令和2年度", "中川", "平和橋付近")],
        [(["軟体動物門", "腹足綱", "新生腹足目", "タニシ科", f"種{i}"],
          [0, 1] if i < 3 else [2]) for i in range(5)],
    )


def test_load_survey_long_form(benthic):
    frame = load_survey(benthic, "benthic")
    assert set(frame.columns) >= {"survey_year", "river", "site", "survey_type", "species"}
    assert frame["survey_type"].unique().tolist() == ["benthic"]
    assert set(frame["survey_year"]) == {1998, 2020}


def test_merged_year_and_river_headers_are_filled(benthic):
    """Years and rivers are written once over a span of merged cells."""
    frame = load_survey(benthic)
    tone = frame[frame.site == "平和橋付近"]
    assert tone["river"].unique().tolist() == ["中川"]
    assert tone["survey_year"].unique().tolist() == [2020]


def test_richness_counts_distinct_taxa(benthic):
    counts = richness(load_survey(benthic, "benthic"))
    assert len(counts) == 3
    at_takumi = counts[counts.site == "内匠橋付近"]
    assert set(at_takumi["richness"]) == {3}
    assert counts[counts.site == "平和橋付近"]["richness"].iloc[0] == 2


def test_richness_on_empty_frame():
    assert richness(pd.DataFrame()).empty


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_survey(tmp_path / "absent.xls")


def test_header_only_workbook_raises(tmp_path):
    path = build_workbook(tmp_path / "empty.xls", BENTHIC_RANKS,
                          [("令和2年度", "綾瀬川", "内匠橋付近")], [])
    with pytest.raises(ValueError, match="no data rows"):
        load_survey(path)


def test_load_all_labels_survey_type(tmp_path):
    build_workbook(tmp_path / "gyo_x.xls", FISH_RANKS,
                   [("令和6年度", "綾瀬川", "内匠橋付近")],
                   [(["コイ目", "コイ科", "コイ"], [0])])
    build_workbook(tmp_path / "teisei_x.xls", BENTHIC_RANKS,
                   [("令和2年度", "綾瀬川", "内匠橋付近")],
                   [(["環形動物門", "ミミズ綱", "目", "科", "イトミミズ"], [0])])
    frame = load_all(tmp_path)
    assert set(frame["survey_type"]) == {"fish", "benthic"}


def test_load_all_empty_directory(tmp_path):
    assert load_all(tmp_path).empty
