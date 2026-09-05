"""Water quality loader tests.

Builds a synthetic workbook that reproduces the published 検体値 schema - seven
header rows, direct fields, then repeating 符合/値 pairs - so the tests do not
need the real 360 KB file or a network call.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.loader import (
    apply_censoring,
    discover_parameters,
    filter_stations,
    load_many,
    load_water_quality,
)

# (row 2 label, row 1 name, row 3 unit, row 6 detection limit)
DIRECT = [
    "測定機関", "水域名", "地点名", "水域", "地点", "西暦年", "月日", "時分",
    "調査区分", "採取位置", "天候", "流況", "臭気コード", "色相", "気温",
    "水温", "流量", "採取水深", "全水深",
]
PARAMS = [("pH", "", ""), ("DO", "mg/L", "0.5"), ("BOD", "mg/L", "0.5"),
          ("全窒素", "mg/L", "0.05"), ("全りん", "mg/L", "0.003")]


def build_workbook(path, rows):
    """rows: list of dicts with direct values + {param: (flag, value)}."""
    width = len(DIRECT) + 2 * len(PARAMS)
    grid = [[np.nan] * width for _ in range(7)]

    grid[0][0] = "測定機関"
    grid[0][len(DIRECT)] = "生活環境項目"

    for i, label in enumerate(DIRECT):
        grid[2][i] = label
    grid[1][1], grid[1][2] = "水域名", "地点名"
    grid[3][0] = "単位"
    grid[6][0] = "2024"

    for j, (name, unit, limit) in enumerate(PARAMS):
        f = len(DIRECT) + 2 * j
        grid[1][f] = name
        grid[2][f], grid[2][f + 1] = "符合", "値"
        grid[3][f + 1] = unit
        grid[6][f + 1] = limit

    body = []
    for r in rows:
        line = [np.nan] * width
        for i, label in enumerate(DIRECT):
            line[i] = r.get(label, np.nan)
        for j, (name, _, _) in enumerate(PARAMS):
            f = len(DIRECT) + 2 * j
            flag, value = r.get(name, (np.nan, np.nan))
            line[f], line[f + 1] = flag, value
        body.append(line)

    pd.DataFrame(grid + body).to_excel(path, header=False, index=False)
    return path


def sample_row(**over):
    row = {
        "測定機関": "埼玉県", "水域名": "綾瀬川下流", "地点名": "52内匠橋",
        "西暦年": 2024, "月日": 417, "時分": 1700,
        "天候": "02晴れ", "流況": "00通常の状況",
        "気温": 18.0, "水温": 21.0, "流量": 22.58,
        "pH": (np.nan, 7.6), "DO": (np.nan, 8.3), "BOD": (np.nan, 4.2),
        "全窒素": (np.nan, 3.4), "全りん": (np.nan, 0.22),
    }
    row.update(over)
    return row


@pytest.fixture
def workbook(tmp_path):
    rows = [
        sample_row(),
        sample_row(月日=508, 時分=730, 水温=20.5, 流量=64.43, DO=(np.nan, 5.6)),
        sample_row(月日=703, 時分=550, 水温=26.1, 流量=50.05, DO=(np.nan, 3.6)),
        sample_row(水域名="中川上流", 地点名="47弥生橋", 月日=812,
                   全りん=("<", 0.003), BOD=("<", 0.5)),
    ]
    return build_workbook(tmp_path / "wq.xlsx", rows)


# ---------------------------------------------------------------------------


def test_loads_expected_shape_and_names(workbook):
    df = load_water_quality(workbook)
    assert len(df) == 4
    for col in ("station", "water_body", "water_temp", "discharge",
                "dissolved_oxygen", "ph", "nitrogen_total", "phosphorus_total"):
        assert col in df.columns


def test_timestamp_assembly(workbook):
    df = load_water_quality(workbook)
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-04-17 17:00")
    assert df["timestamp"].iloc[1] == pd.Timestamp("2024-05-08 07:30")


def test_missing_time_falls_back_to_midnight(tmp_path):
    wb = build_workbook(tmp_path / "t.xlsx", [sample_row(時分=np.nan)])
    df = load_water_quality(wb)
    assert len(df) == 1
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-04-17 00:00")


def test_rows_without_a_date_are_dropped(tmp_path):
    wb = build_workbook(tmp_path / "t.xlsx", [sample_row(), sample_row(月日=np.nan)])
    assert len(load_water_quality(wb)) == 1


def test_code_prefixes_stripped(workbook):
    df = load_water_quality(workbook)
    assert df["weather"].iloc[0] == "晴れ"
    assert df["flow_condition"].iloc[0] == "通常の状況"


def test_numeric_coercion(workbook):
    df = load_water_quality(workbook)
    assert df["water_temp"].dtype.kind == "f"
    assert df["discharge"].iloc[2] == pytest.approx(50.05)
    assert df["dissolved_oxygen"].iloc[2] == pytest.approx(3.6)


def test_units_and_limits_recorded(workbook):
    df = load_water_quality(workbook)
    assert df.attrs["units"]["dissolved_oxygen"] == "mg/L"
    assert df.attrs["detection_limits"]["phosphorus_total"] == pytest.approx(0.003)


def test_parameter_discovery_pairs_flag_with_value(workbook):
    header = pd.read_excel(workbook, header=None).iloc[:7]
    params = discover_parameters(header)
    assert [p.column for p in params] == [
        "ph", "dissolved_oxygen", "bod", "nitrogen_total", "phosphorus_total"
    ]
    assert all(p.value_index == p.flag_index + 1 for p in params)


def test_parameter_subset(workbook):
    df = load_water_quality(workbook, parameters=["dissolved_oxygen"])
    assert "dissolved_oxygen" in df.columns
    assert "bod" not in df.columns


# --- censoring --------------------------------------------------------------


def test_flags_preserved_and_value_untouched_by_default(workbook):
    df = load_water_quality(workbook)
    assert df["phosphorus_total_flag"].iloc[3] == "<"
    assert df["phosphorus_total"].iloc[3] == pytest.approx(0.003)
    assert pd.isna(df["phosphorus_total_flag"].iloc[0])


def test_half_policy_halves_only_censored_rows(workbook):
    df = apply_censoring(load_water_quality(workbook), policy="half")
    assert df["phosphorus_total"].iloc[3] == pytest.approx(0.0015)
    assert df["phosphorus_total"].iloc[0] == pytest.approx(0.22)  # untouched


def test_nan_and_zero_policies(workbook):
    base = load_water_quality(workbook)
    assert pd.isna(apply_censoring(base, policy="nan")["bod"].iloc[3])
    assert apply_censoring(base, policy="zero")["bod"].iloc[3] == 0.0


def test_keep_policy_is_identity(workbook):
    base = load_water_quality(workbook)
    assert apply_censoring(base, policy="keep")["bod"].iloc[3] == pytest.approx(0.5)


def test_over_range_flag_never_substituted(tmp_path):
    wb = build_workbook(tmp_path / "o.xlsx", [sample_row(DO=(">", 20.0))])
    df = apply_censoring(load_water_quality(wb), policy="half")
    assert df["dissolved_oxygen"].iloc[0] == pytest.approx(20.0)


# --- filtering and concatenation -------------------------------------------


def test_filter_by_water_body_and_station(workbook):
    df = load_water_quality(workbook)
    assert len(filter_stations(df, water_body="綾瀬川")) == 3
    assert len(filter_stations(df, station="弥生橋")) == 1
    assert len(filter_stations(df, water_body="中川", station="弥生橋")) == 1


def test_load_many_sorts_and_merges_metadata(tmp_path):
    a = build_workbook(tmp_path / "a.xlsx", [sample_row(月日=1201)])
    b = build_workbook(tmp_path / "b.xlsx", [sample_row(月日=105)])
    df = load_many([a, b])
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing
    assert "dissolved_oxygen" in df.attrs["units"]


def test_load_many_empty():
    assert load_many([]).empty


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_water_quality(tmp_path / "nope.xlsx")


def test_header_only_workbook_raises(tmp_path):
    wb = build_workbook(tmp_path / "empty.xlsx", [])
    with pytest.raises(ValueError, match="no observations"):
        load_water_quality(wb)
