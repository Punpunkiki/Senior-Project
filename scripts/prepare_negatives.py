"""
prepare_negatives.py — build a balanced, durian-free `not_durian` class from a
dataset that has one sub-folder per fruit type (e.g. Fruits-262).

Why this script and not just point the pipeline at the folder:
  - The pipeline reads nested folders recursively, so it WOULD pull every image
    in as `not_durian` — but that (a) silently includes any `durian/` sub-folder
    (label poisoning) and (b) dumps ~hundreds of thousands of images in, dwarfing
    the ~400/disease classes (severe imbalance).
  - This script fixes both: it EXCLUDES durian-like folders, samples images
    EVENLY across fruit types (so the negative class is diverse, not 90% apples),
    caps the total to a target count, and optionally HOLDS OUT whole fruit types
    as an unseen-negative test set (the honest way to measure open-set
    generalisation — [DL-CLASSES]).

Examples:
  # 600 balanced negatives into data/not_durian, durian excluded:
  python scripts/prepare_negatives.py --src /path/fruits-262 --dest data/not_durian --n 600

  # also reserve ~20% of fruit TYPES as an unseen held-out negative set:
  python scripts/prepare_negatives.py --src /path/fruits-262 --dest data/not_durian \
      --n 600 --holdout-frac 0.2 --holdout-dest data/not_durian_heldout
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger  # noqa: E402

log = get_logger("prep_neg")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
DEFAULT_EXCLUDE = ["durian"]   # keywords -> skip that fruit-type folder


def _list_types(src: Path) -> Dict[str, List[Path]]:
    """Return {type_name: [image paths]} from immediate sub-folders; if there are
    no sub-folders, treat all images in src as a single '_root' type."""
    subdirs = [p for p in sorted(src.iterdir()) if p.is_dir()]
    types: Dict[str, List[Path]] = {}
    if subdirs:
        for d in subdirs:
            imgs = [p for p in d.rglob("*") if p.suffix in IMG_EXTS]
            if imgs:
                types[d.name] = imgs
    else:
        imgs = [p for p in src.rglob("*") if p.suffix in IMG_EXTS]
        if imgs:
            types["_root"] = imgs
    return types


def _excluded(name: str, keywords: List[str]) -> bool:
    low = name.lower()
    return any(k.lower() in low for k in keywords)


def _round_robin(pools: Dict[str, List[Path]], n: int, rng) -> List[tuple]:
    """Pick up to n items, one per type per round (even diversity)."""
    order = list(pools.keys())
    for k in order:
        rng.shuffle(pools[k])
    picked, idx = [], {k: 0 for k in order}
    while len(picked) < n:
        progressed = False
        for k in order:
            if idx[k] < len(pools[k]):
                picked.append((k, pools[k][idx[k]])); idx[k] += 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:   # all pools exhausted
            break
    return picked


def _copy_as_jpg(src_path: Path, dest_path: Path, symlink: bool) -> bool:
    try:
        if symlink:
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            os.symlink(src_path.resolve(), dest_path)
        else:
            from PIL import Image
            with Image.open(src_path) as im:
                im.convert("RGB").save(dest_path, "JPEG", quality=90)
        return True
    except Exception as e:
        log.warning("skip %s (%s)", src_path, e)
        return False


def _write(picked: List[tuple], dest: Path, symlink: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for tname, p in picked:
        safe = f"{tname}__{p.stem}".replace(os.sep, "_").replace(" ", "_")
        out = dest / f"{safe}.jpg"
        i = 1
        while out.exists():            # avoid collisions
            out = dest / f"{safe}_{i}.jpg"; i += 1
        if _copy_as_jpg(p, out, symlink):
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with per-fruit subfolders")
    ap.add_argument("--dest", default="data/not_durian", help="flat output folder")
    ap.add_argument("--n", type=int, default=600, help="target #images for not_durian")
    ap.add_argument("--per-type-cap", type=int, default=0,
                    help="max images taken per fruit type (0 = no cap)")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="skip fruit-type folders whose name contains these")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="fraction of TYPES reserved as unseen negatives")
    ap.add_argument("--holdout-dest", default="data/not_durian_heldout")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink instead of copy+convert (keeps original format)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)
    src = Path(args.src)
    if not src.is_dir():
        log.error("src not found: %s", src); sys.exit(2)

    types = _list_types(src)
    if not types:
        log.error("no images found under %s", src); sys.exit(2)

    excluded = [t for t in types if _excluded(t, args.exclude)]
    for t in excluded:
        del types[t]
    if excluded:
        log.info("EXCLUDED %d type(s) by keyword %s: %s",
                 len(excluded), args.exclude, excluded)
    log.info("Found %d usable fruit type(s), %d candidate images.",
             len(types), sum(len(v) for v in types.values()))

    # optional: hold out whole TYPES as unseen negatives
    held = {}
    if args.holdout_frac > 0 and len(types) > 1:
        names = list(types.keys()); rng.shuffle(names)
        n_hold = max(1, round(args.holdout_frac * len(names)))
        for t in names[:n_hold]:
            held[t] = types.pop(t)
        log.info("Held out %d type(s) as unseen negatives: %s",
                 len(held), sorted(held))

    if args.per_type_cap > 0:
        for t in types:
            if len(types[t]) > args.per_type_cap:
                rng.shuffle(types[t]); types[t] = types[t][:args.per_type_cap]

    picked = _round_robin(types, args.n, rng)
    n_main = _write(picked, Path(args.dest), args.symlink)
    log.info("Wrote %d images -> %s (from %d types).",
             n_main, args.dest, len({k for k, _ in picked}))

    if held:
        held_picked = [(t, p) for t, ps in held.items() for p in ps]
        n_hold = _write(held_picked, Path(args.holdout_dest), args.symlink)
        log.info("Wrote %d held-out unseen negatives -> %s (types: %s).",
                 n_hold, args.holdout_dest, sorted(held))
        log.info("Use the held-out set to report open-set generalisation: train "
                 "with data/not_durian, then measure reject behaviour on it.")

    log.info("DONE. Tip: keep ~similar counts to your ~400/disease classes; "
             "current not_durian = %d.", n_main)


if __name__ == "__main__":
    main()
