"""Centralised configuration for AquaNexus.

Settings are read from environment variables (or a local ``.env``) and fall back
to the defaults below. Import the module-level ``settings`` singleton rather
than instantiating ``Settings`` yourself, so path resolution happens once.
"""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Project ---
    PROJECT_NAME: str = "AquaNexus"
    VERSION: str = "0.1.0"

    # --- Paths ---
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    HECRAS_DIR: Path = DATA_DIR / "hecras"
    CACHE_DIR: Path = DATA_DIR / "cache"

    # --- Study site ---
    # Selected 2026-09-06; see docs/DATA_SOURCES.md for the availability audit
    # that ruled out the Yodo River and settled on the Saitama rivers.
    RIVER_NAME: str = "Ayase"
    RIVER_NAME_JA: str = "綾瀬川"
    RIVER_SYSTEM: str = "Tone"
    # JGD2011 / Japan Plane Rectangular CS Zone 9 — matches the Saitama
    # point-cloud release. Do not change without reprojecting the geometry.
    CRS_EPSG: int = 6677

    # --- HEC-RAS ---
    # Verified against HEC-RAS 7.0 (April 2026) via its COM controller. If this
    # path is wrong, runner.find_hecras_exe() falls back to scanning the install
    # root, so a version bump does not break the pipeline.
    HECRAS_EXE: Path = Path(r"C:\Program Files (x86)\HEC\HEC-RAS\7.0\Ras.exe")
    HECRAS_VERSION: str = "7.00"  # written into generated .g01 headers
    HECRAS_TIMEOUT: int = 600  # seconds per simulation run

    # --- ML ---
    MODEL_TYPE: Literal["xgboost", "random_forest", "linear", "lstm"] = "xgboost"
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- Logging ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False

    @field_validator("TEST_SIZE", "VAL_SIZE")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("split fractions must lie strictly between 0 and 1")
        return v

    def ensure_dirs(self) -> None:
        """Create the data directories if they are missing.

        Called explicitly by scripts rather than at import time, so that merely
        importing the config never touches the filesystem.
        """
        for path in (
            self.RAW_DIR,
            self.PROCESSED_DIR,
            self.MODELS_DIR,
            self.HECRAS_DIR,
            self.CACHE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
