"""Scaffold-level checks: configuration resolves and the package imports cleanly."""

from pathlib import Path

import pytest

from aquanexus import __version__
from aquanexus.config import Settings, settings
from aquanexus.logger import get_logger


def test_version_matches_settings():
    assert __version__ == settings.VERSION


def test_root_dir_is_repo_root():
    # config.py lives at <root>/src/aquanexus/config.py
    assert (settings.ROOT_DIR / "pyproject.toml").is_file()


def test_data_paths_are_under_data_dir():
    for path in (
        settings.RAW_DIR,
        settings.PROCESSED_DIR,
        settings.MODELS_DIR,
        settings.HECRAS_DIR,
        settings.CACHE_DIR,
    ):
        assert settings.DATA_DIR in path.parents


def test_ensure_dirs_is_idempotent(tmp_path: Path):
    s = Settings(DATA_DIR=tmp_path, RAW_DIR=tmp_path / "raw", PROCESSED_DIR=tmp_path / "proc",
                 MODELS_DIR=tmp_path / "models", HECRAS_DIR=tmp_path / "hecras",
                 CACHE_DIR=tmp_path / "cache")
    s.ensure_dirs()
    s.ensure_dirs()
    assert (tmp_path / "raw").is_dir()


@pytest.mark.parametrize("field", ["TEST_SIZE", "VAL_SIZE"])
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_split_fractions_rejected_outside_unit_interval(field, bad):
    with pytest.raises(ValueError):
        Settings(**{field: bad})


def test_crs_is_jgd2011_zone_9():
    # The Saitama point cloud ships in this CRS; changing it silently would
    # misplace every cross-section.
    assert settings.CRS_EPSG == 6677


def test_logger_is_namespaced():
    assert get_logger("hecras.reader").name == "aquanexus.hecras.reader"


def test_logger_does_not_duplicate_handlers():
    """Repeated get_logger calls must not stack handlers.

    Counted as a delta rather than an absolute, because pytest attaches its own
    capture handlers to the same logger.
    """
    import logging

    get_logger("warmup")
    before = len(logging.getLogger("aquanexus").handlers)
    get_logger("a")
    get_logger("b")
    assert len(logging.getLogger("aquanexus").handlers) == before
