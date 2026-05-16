"""Download the OULAD release into data/raw.

The public OULAD CSV bundle is available at:
  https://analyse.kmi.open.ac.uk/open_dataset/download

This script downloads and unpacks the archive when network access is
permitted. Otherwise the user can drop the seven CSVs manually into
data/raw/.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="data/raw")
    args = parser.parse_args()
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    archive = target / "oulad.zip"
    print(f"downloading {OULAD_URL} -> {archive}")
    try:
        with urlopen(OULAD_URL, timeout=60) as resp, archive.open("wb") as fh:
            fh.write(resp.read())
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print("please drop the OULAD CSVs into data/raw manually.")
        return

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink(missing_ok=True)
    print("done. Files extracted into data/raw/")


if __name__ == "__main__":
    main()
