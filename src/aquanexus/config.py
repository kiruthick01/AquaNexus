"""Centralised configuration for AquaNexus.

Settings are read from environment variables (or a local ``.env``) and fall back
to the defaults below. Import the module-level ``settings`` singleton rather
than instantiating ``Settings`` yourself, so path resolution happens once.
"""

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
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
    # /explain costs ~190 ms of SHAP per call. The cache makes a repeated state
    # free; the rate limit caps how fast the expensive path can be entered.
    # Both are per-process: with N uvicorn workers the effective ceiling is
    # N x EXPLAIN_RATE_LIMIT. See docs/DEPLOYMENT.md.
    EXPLAIN_RATE_LIMIT: int = 60  # requests per minute per client
    EXPLAIN_CACHE_SIZE: int = 256
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    # Both spellings of the dev origins: a browser treats http://127.0.0.1:3000
    # and http://localhost:3000 as different origins, and a request from the
    # unlisted one fails as an unexplained "cannot reach the API".
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ]

    # --- Logging ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False

    @field_validator("TEST_SIZE", "VAL_SIZE")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("split fractions must lie strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _keep_data_paths_under_data_dir(self):
        """Re-derive the data subdirectories whenever DATA_DIR is overridden.

        ``ROOT_DIR`` is computed from this file's location, which is correct for
        a checkout and wrong for an installed package: in a container the code
        lives in site-packages while the data is mounted at /app/data. Setting
        ``DATA_DIR`` must therefore move everything under it - the subdirectory
        defaults were frozen against the *default* DATA_DIR when the class was
        defined, so without this the override silently applies to DATA_DIR alone
        and the API starts degraded, looking for models beside its own source.

        Explicitly set subdirectories are left alone.
        """
        if "DATA_DIR" not in self.model_fields_set:
            return self
        for name, child in (("RAW_DIR", "raw"), ("PROCESSED_DIR", "processed"),
                            ("MODELS_DIR", "models"), ("HECRAS_DIR", "hecras"),
                            ("CACHE_DIR", "cache")):
            if name not in self.model_fields_set:
                object.__setattr__(self, name, self.DATA_DIR / child)
        return self

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
