"""Sort MOE 水環境総合情報サイト download zips into ``data/raw/moe/{mk,mm}_<fiscal year>/``.

The site names every download with a random string, so each zip is identified by
its contents: ``ZETTAICODE`` marks a sample file (mk), ``zettaicode`` a station
file (mm), and the ``NENDO``/``nendo`` column gives the fiscal year.

    python scripts/unpack_moe.py ~/Downloads
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

ENCODING = "cp932"


def identify(path: Path) -> tuple[str, str] | None:
    """Return ``(kind, fiscal_year)`` for a MOE zip, or None if it is something else."""
    try:
        with zipfile.ZipFile(path) as z:
            if z.namelist() != ["output.csv"]:
                return None
            reader = csv.reader(io.TextIOWrapper(z.open("output.csv"), encoding=ENCODING))
            header = next(reader)
            if "ZETTAICODE" in header:
                kind, column = "mk", header.index("NENDO")
            elif "zettaicode" in header:
                kind, column = "mm", header.index("nendo")
            else:
                return None
            years = {row[column] for row in reader}
    except (zipfile.BadZipFile, UnicodeDecodeError, ValueError, StopIteration, OSError):
        return None
    return (kind, years.pop()) if len(years) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="folder holding the downloaded zips")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/moe"))
    args = parser.parse_args()

    placed: set[tuple[str, str]] = set()
    for path in sorted(args.source.glob("*.zip")):
        found = identify(path)
        if found is None or found in placed:
            continue
        kind, year = found
        target = args.dest / f"{kind}_{year}"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            (target / "output.csv").write_bytes(z.read("output.csv"))
        placed.add(found)
        print(f"{path.name} -> {target}")
    print(f"{len(placed)} unique files placed under {args.dest}")


if __name__ == "__main__":
    main()
