"""Centralised configuration for AquaNexus.

Settings are read from environment variables (or a local ``.env``) and fall back
to the defaults below. Import the module-level ``settings`` singleton rather
than instantiating ``Settings`` yourself, so path resolution happens once.
"""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
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
    CACHE_DIR: Path = DATA_DIR / "cache"

    # --- API ---
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
                            ("MODELS_DIR", "models"),
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
            self.CACHE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
