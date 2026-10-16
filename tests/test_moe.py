"""Tests for the MOE nationwide sample/station loader."""

from __future__ import annotations

import pandas as pd
import pytest

from aquanexus.data import moe

MK_COLUMNS = [
    "ZETTAICODE", "PREFCODE", "NENDO", "FISCALYEAR", "MONTH", "DAYTIME", "DEPTH",
    "PH", "DO_", "DO_3", "BOD", "COD", "SS",
]  # fmt: skip


def _write_mk(path, rows):
    pd.DataFrame(rows, columns=MK_COLUMNS).to_csv(path, index=False, encoding=moe.ENCODING)


def _write_mm(path, rows):
    columns = ["zettaicode", "nendo", "locationname", "water", "latitude", "longitude"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding=moe.ENCODING)


@pytest.fixture()
def raw_dir(tmp_path):
    mk = tmp_path / "mk_2022"
    mm = tmp_path / "mm_2022"
    mk.mkdir()
    mm.mkdir()
    _write_mk(
        mk / "output.csv",
        [
            # station, pref, fiscal, calendar, month, ddhhmm, depth, pH, DO, DO_3, BOD, COD, SS
            [1, 11, 2022, 2022, 6, 30930, 1, 7.5, "9.4", 0, "1.2", None, "8"],
            [1, 11, 2022, 2023, 3, 221247, 1, 7.1, "5.0", 1, "<0.5", None, "3"],
            [2, 13, 2022, 2022, 11, 101030, 1, 7.9, "<0.5", 1, "2.0", "4.1", "12"],
            [2, 13, 2022, 2022, 12, 101030, 1, 7.9, None, 0, "2.0", None, "12"],
            [3, 13, 2022, 2022, 7, 150900, 1, 7.0, "31.0", 0, "1.0", None, "4"],
            [4, 14, 2022, 2022, 8, 10800, 1, 7.2, "8.0", 0, "1.0", None, "4"],
        ],
    )
    _write_mm(
        mm / "output.csv",
        [
            [1, 2021, "A bridge", "River A", 35.0, 139.0],
            [1, 2022, "A bridge", "River A", 35.1, 139.1],
            [2, 2022, "B bridge", "River B", 36.0, 140.0],
            [3, 2022, "C bridge", "River C", 37.0, 141.0],
        ],
    )
    return tmp_path


def test_daytime_is_unpacked_as_unpadded_ddhhmm():
    out = moe._unpack_daytime(pd.Series([30930, 221247, 101030, 10800]))
    assert out["day"].tolist() == [3, 22, 10, 1]
    assert out["hour"].tolist() == [9, 12, 10, 8]
    assert out["minute"].tolist() == [30, 47, 30, 0]


def test_censored_value_becomes_half_the_limit_and_is_flagged():
    value, censored = moe._parse_censored(pd.Series(["<0.5", "9.4", None]))
    assert value.iloc[0] == pytest.approx(0.25)
    assert value.iloc[1] == pytest.approx(9.4)
    assert pd.isna(value.iloc[2])
    assert censored.tolist() == [True, False, False]


def test_fiscal_and_calendar_year_follow_the_files_odd_naming(raw_dir):
    samples = moe.read_samples(raw_dir / "mk_2022" / "output.csv")
    march = samples[samples["month"] == 3].iloc[0]
    assert march["fiscal_year"] == 2022
    assert march["year"] == 2023
    assert march["date"] == pd.Timestamp("2023-03-22")


def test_below_standard_flag_is_kept_separate_from_features(raw_dir):
    samples = moe.read_samples(raw_dir / "mk_2022" / "output.csv")
    assert samples["below_standard"].tolist() == [False, True, True, False, False, False]


def test_stations_keep_one_row_with_latest_coordinate(raw_dir):
    stations = moe.load_stations(moe.find_files(raw_dir, "mm"))
    assert stations["station_id"].is_unique
    a = stations.set_index("station_id").loc[1]
    assert (a["latitude"], a["longitude"]) == (35.1, 139.1)


def test_dataset_drops_unusable_rows_and_keeps_censored_do(raw_dir):
    samples = moe.load_samples(moe.find_files(raw_dir, "mk"))
    stations = moe.load_stations(moe.find_files(raw_dir, "mm"))
    data = moe.build_dataset(samples, stations)

    # missing DO (row 4), DO above 25 mg/L (row 5), and no coordinates (station 4) are dropped
    assert len(data) == 3
    assert set(data["station_id"]) == {1, 2}
    assert data["dissolved_oxygen"].between(*moe.DO_RANGE).all()
    assert data["latitude"].notna().all()
    censored = data[data["dissolved_oxygen_censored"]]
    assert censored["dissolved_oxygen"].tolist() == [0.25]
