"""
download_data.py — fetch the Mendeley durian dataset into ./data/.

  Dataset: "Image Dataset of Ten Durian Diseases Captured in Real-Field
            Conditions from a Family Orchard in Vinh Long, Vietnam"
  Mendeley DOI: 10.17632/mhjwyb5p48.1
  Landing page: https://data.mendeley.com/datasets/mhjwyb5p48/1

NOTE: run this on a machine with open internet. Some sandboxes (including the
one this repo may have been generated in) block data.mendeley.com by policy —
in that case download the ZIP manually from the landing page and unzip into
./data/ so it ends up with 10 class sub-folders.

Run:  python -m src.download_data            # confirms before downloading
      python -m src.download_data --yes       # non-interactive
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from .utils import ensure_dir, get_logger

log = get_logger("download")

DATASET_ID = "mhjwyb5p48"
VERSION = 1
API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?version={VERSION}"


def fetch_file_list():
    import requests
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return r.json()


def download(dest: Path, yes: bool = False) -> None:
    import requests
    ensure_dir(dest)
    try:
        files = fetch_file_list()
    except Exception as e:
        log.error("Could not reach Mendeley API (%s). Download the ZIP manually "
                  "from https://data.mendeley.com/datasets/%s/%d and unzip into "
                  "%s", e, DATASET_ID, VERSION, dest)
        sys.exit(2)

    total_mb = sum(f.get("size", 0) for f in files) / 1e6
    log.info("Dataset has %d file(s), ~%.0f MB total.", len(files), total_mb)
    if not yes:
        ans = input(f"Download ~{total_mb:.0f} MB into {dest}? [y/N] ").strip().lower()
        if ans != "y":
            log.info("Aborted by user.")
            return

    for f in files:
        name = f["filename"]
        url = f["content_details"]["download_url"]
        target = dest / name
        log.info("Downloading %s ...", name)
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(target, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        if target.suffix.lower() == ".zip":
            log.info("Unzipping %s ...", name)
            with zipfile.ZipFile(target) as z:
                z.extractall(dest)
    log.info("Done. Verify ./data now contains 10 class folders, then run "
             "the EDA notebook to confirm real per-class counts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./data")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    download(Path(args.dest), yes=args.yes)


if __name__ == "__main__":
    main()
