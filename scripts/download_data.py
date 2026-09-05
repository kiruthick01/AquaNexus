"""Fetch open datasets for the study site.

Phase 1b will implement the Saitama point-cloud and MOE water-quality fetchers.

NOTE: MLIT's 水文水質データベース (river.go.jp) prohibits automated acquisition.
Discharge and stage are downloaded by hand through its web UI (30 days per request)
and dropped into data/raw/. This script must not attempt to fetch from it.
"""

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("scripts.download_data")


def main() -> None:
    settings.ensure_dirs()
    log.info("Target site: %s (%s)", settings.RIVER_NAME, settings.RIVER_NAME_JA)
    log.warning("Not implemented yet — see docs/DATA_SOURCES.md for manual steps.")


if __name__ == "__main__":
    main()
