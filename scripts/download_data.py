"""Fetch the open datasets for the study site.

    python scripts/download_data.py                # water quality (default)
    python scripts/download_data.py --tile-index   # refresh the point cloud index

Point-cloud tiles themselves are downloaded by ``scripts/build_geometry.py``,
which needs to know which river and how many.

NOT HANDLED HERE - and deliberately so
--------------------------------------
MLIT's 水文水質データベース (river.go.jp) answers programmatic requests with
"This site prohibits data acquisition using tools". Hourly discharge and stage
for boundary conditions must be downloaded by hand through its web UI, 30 days
per request. Do not add a fetcher for it.

The monthly water quality files below already carry co-measured discharge and
water temperature, which covers the observational anchor without that manual work.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("scripts.download_data")

USER_AGENT = "AquaNexus/0.1 (research; +https://github.com/kiruthick01/AquaNexus)"

# Saitama Prefecture 公共用水域水質測定データ (検体値). Reiwa year -> fiscal year.
SAITAMA_WQ = {
    2022: "https://www.pref.saitama.lg.jp/documents/15286/r04_suishitsu_data.xlsx",
    2023: "https://www.pref.saitama.lg.jp/documents/15286/r05_suishitsu_data.xlsx",
    2024: "https://www.pref.saitama.lg.jp/documents/15286/r06_suishitsu_data.xlsx",
}


def fetch(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.is_file() and not overwrite:
        log.info("%s already present (%d KB)", dest.name, dest.stat().st_size // 1024)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    log.info("wrote %s (%d KB)", dest.name, dest.stat().st_size // 1024)
    return dest


def download_water_quality(overwrite: bool = False) -> list[Path]:
    out_dir = settings.RAW_DIR / "waterquality"
    written = []
    for year, url in sorted(SAITAMA_WQ.items()):
        try:
            written.append(fetch(url, out_dir / f"saitama_{year}.xlsx", overwrite))
        except Exception as exc:
            log.error("FY%d failed: %s", year, exc)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tile-index", action="store_true",
                    help="refresh the cached point-cloud tile index")
    ap.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    settings.ensure_dirs()
    log.info("site: %s (%s)", settings.RIVER_NAME, settings.RIVER_NAME_JA)

    if args.tile_index:
        from aquanexus.data.pointcloud import load_tile_index

        tiles = load_tile_index(refresh=True)
        rivers = {t.river for t in tiles}
        log.info("tile index: %d tiles across %d rivers", len(tiles), len(rivers))
        return 0

    files = download_water_quality(args.overwrite)
    if not files:
        log.error("nothing downloaded")
        return 1

    log.info("%d water quality file(s) in %s", len(files), settings.RAW_DIR / "waterquality")
    log.warning(
        "Hourly discharge for HEC-RAS boundary conditions still needs a manual "
        "pull from river.go.jp - see the module docstring."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
