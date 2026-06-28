"""
data.py — dataset discovery, leakage-safe splitting, transforms, dataloaders.

Pipeline:
  1. discover_images()  : map on-disk folders -> canonical labels, pick jpeg/png.
  2. assign_groups()    : perceptual-hash near-duplicate clustering ([DL-GROUP]).
  3. make_splits()      : group-aware stratified train/val/test ([DL-SPLIT]).
  4. build_transforms() : surgical augmentation ([DL-AUG-ON]/[DL-AUG-OFF]).
  5. build_dataloaders(): ready-to-train loaders (+ quarantined OOD, [DL-OOD]).

Everything tunable is read from config.yaml; no magic numbers here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .utils import get_logger, save_json, seed_worker

log = get_logger("data")

IMG_EXT = {
    "jpeg": (".jpg", ".jpeg", ".JPG", ".JPEG"),
    "png": (".png", ".PNG"),
    # 'any' accepts every common format -> use it if Healthy/not_durian are PNG
    # etc. (set data.image_format: any in config.yaml).
    "any": (".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"),
}


# --------------------------------------------------------------------------- #
# 1. Discovery
# --------------------------------------------------------------------------- #
def _match_canonical(folder_name: str, aliases: Dict[str, List[str]]) -> Optional[str]:
    low = folder_name.lower().replace("-", " ").replace("_", " ").strip()
    best, best_len = None, -1
    for canon, keys in aliases.items():
        for k in keys:
            if k.lower() in low and len(k) > best_len:
                best, best_len = canon, len(k)
    return best


def _scan_class_dirs(root: Path, exts, aliases, label_to_idx) -> tuple:
    """Scan one root for <class>/ sub-folders -> (rows, unmatched_names)."""
    rows, unmatched = [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        canon = _match_canonical(sub.name, aliases)
        if canon is None:
            unmatched.append(sub.name)
            continue
        for fp in sub.rglob("*"):           # .DS_Store etc. ignored by suffix filter
            if fp.suffix in exts:
                rows.append({"path": str(fp), "folder": sub.name,
                             "label": canon, "label_idx": label_to_idx[canon]})
    return rows, unmatched


def discover_images(cfg: Dict[str, Any], data_root: Optional[str] = None,
                    image_format: Optional[str] = None) -> pd.DataFrame:
    """Flat layout: data_root/<class>/...  ->  DataFrame[path, folder, label,
    label_idx].  Reports the TRUTH about counts/format mismatches vs the proposal.
    """
    root = Path(data_root or cfg["paths"]["data_root"])
    fmt = (image_format or cfg["data"]["image_format"]).lower()
    if not root.exists():
        raise FileNotFoundError(
            f"data_root '{root}' not found. Provide the dataset so it contains the "
            f"class sub-folders (see README / src/download_data.py)."
        )
    exts = IMG_EXT[fmt]
    classes = cfg["data"]["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    rows, unmatched = _scan_class_dirs(root, exts, cfg["data"]["folder_aliases"],
                                       label_to_idx)
    if unmatched:
        log.warning("Folders not matched to a canonical class: %s", unmatched)
    if not rows:
        raise RuntimeError(
            f"No '{fmt}' images found under {root}. Check config.data.image_format "
            f"(jpeg vs png) and the folder layout."
        )
    df = pd.DataFrame(rows)
    _report_counts(df, classes, fmt)
    return df


def discover_predefined_split(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Pre-split layout: data_root/<SplitDir>/<class>/...  e.g.
    Train/Validation/Test (the user's Google-Drive structure). Returns a
    DataFrame with the 'split' column already set from the folder names.
    ([DL-PRESPLIT])
    """
    root = Path(cfg["paths"]["data_root"])
    fmt = cfg["data"]["image_format"].lower()
    exts = IMG_EXT[fmt]
    classes = cfg["data"]["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    dirmap = cfg["split"]["predefined_dirs"]   # {train: Train, val: Validation, ...}

    frames = []
    for split_name, dirname in dirmap.items():
        sdir = root / dirname
        if not sdir.exists():
            raise FileNotFoundError(
                f"Predefined-split dir '{sdir}' not found. Expected "
                f"{root}/{{{', '.join(dirmap.values())}}}/<class>/. "
                f"Set split.predefined_dirs in config.yaml to match your folders."
            )
        rows, unmatched = _scan_class_dirs(sdir, exts, cfg["data"]["folder_aliases"],
                                           label_to_idx)
        if unmatched:
            log.warning("[%s] folders not matched to a class: %s", dirname, unmatched)
        f = pd.DataFrame(rows)
        f["split"] = split_name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"No '{fmt}' images found under predefined split dirs.")

    # Convenience: auto-split any FLAT extra-class folders at data_root (e.g. a
    # freshly-added data/Healthy or data/not_durian that isn't pre-split). They
    # get a group-aware stratified split and are merged into Train/Val/Test.
    extra = _discover_extra_flat_classes(cfg, root, exts, dirmap, label_to_idx)
    if extra is not None and len(extra):
        extra = _autosplit_extra(cfg, extra)
        df = pd.concat([df, extra[df.columns]], ignore_index=True)
        log.info("Merged %d auto-split extra-class image(s) from flat folders: %s",
                 len(extra), sorted(extra["label"].unique()))

    _report_counts(df, classes, fmt)
    _report_split(df, classes)
    return df


def _discover_extra_flat_classes(cfg, root, exts, dirmap, label_to_idx):
    """Find class folders sitting flat at data_root (NOT inside the predefined
    split dirs) so they can be auto-split and merged in."""
    split_dirs = {v.lower() for v in dirmap.values()}
    rows = []
    for sub in (p for p in root.iterdir() if p.is_dir()):
        if sub.name.lower() in split_dirs:
            continue
        canon = _match_canonical(sub.name, cfg["data"]["folder_aliases"])
        if canon is None:
            continue
        for fp in sub.rglob("*"):
            if fp.suffix in exts:
                rows.append({"path": str(fp), "folder": sub.name,
                             "label": canon, "label_idx": label_to_idx[canon]})
    return pd.DataFrame(rows) if rows else None


def _autosplit_extra(cfg, extra: pd.DataFrame) -> pd.DataFrame:
    """Group-aware stratified split of the flat extra classes, using the ratios
    in config.split. Falls back to plain stratified if groups are too few."""
    import copy
    sub = copy.deepcopy(cfg)
    extra = assign_groups(cfg, extra)
    for method in ("group_stratified", "stratified", "random"):
        try:
            sub["split"]["method"] = method
            out = make_splits(sub, extra, cfg["seed"])
            if method != "group_stratified":
                log.warning("Extra-class auto-split fell back to '%s' (too few "
                            "groups for group-aware split).", method)
            return out
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not auto-split extra classes: {last}")


def audit_split_leakage(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """For a PREDEFINED split we cannot enforce group-awareness, so we AUDIT it:
    detect near-duplicates (pHash) that straddle train/val/test. A non-zero count
    is a leakage warning the report must disclose ([DL-PRESPLIT]).

    Returns (report_dict, grouped_df) where grouped_df has a 'group' column.
    """
    if "split" not in df.columns:
        return {}, df
    grouped = df if "group" in df.columns else assign_groups(cfg, df)
    span = grouped.groupby("group")["split"].nunique()
    leaky_groups = span[span > 1]
    leaky_imgs = int(grouped["group"].isin(leaky_groups.index).sum())
    rep = {"n_cross_split_dup_groups": int(len(leaky_groups)),
           "n_images_in_cross_split_groups": leaky_imgs,
           "note": ("Near-duplicates spanning splits inflate the test score. "
                    "Predefined split kept for comparability; disclose this number "
                    "and consider de-duping or a group-aware re-split.")}
    if len(leaky_groups):
        log.warning("LEAKAGE AUDIT: %d near-duplicate group(s) span splits "
                    "(%d images). See [DL-PRESPLIT].", len(leaky_groups), leaky_imgs)
    else:
        log.info("LEAKAGE AUDIT: no cross-split near-duplicates detected.")
    return rep, grouped


def load_full_dataframe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return the complete image DataFrame regardless of split mode (predefined
    rows carry a 'split' column). Used by EDA so it works either way."""
    if cfg["split"]["method"] == "predefined":
        return discover_predefined_split(cfg)
    return discover_images(cfg)


def _report_counts(df: pd.DataFrame, classes: List[str], fmt: str) -> None:
    counts = df["label"].value_counts().reindex(classes).fillna(0).astype(int)
    log.info("Image format used: %s | total images: %d", fmt, len(df))
    log.info("Per-class counts (TRUTH, not the proposal's claim):")
    for c in classes:
        log.info("  %-26s %4d", c, counts[c])
    imbalance = counts.max() / max(counts.min(), 1)
    log.info("max/min class ratio = %.2f (proposal claimed perfectly balanced)",
             imbalance)
    missing = [c for c in classes if counts[c] == 0]
    if missing:
        log.warning("These classes have 0 '%s' images: %s. If the files exist in "
                    "a different format, set data.image_format: any (or png) in "
                    "config.yaml.", fmt, missing)


# --------------------------------------------------------------------------- #
# 2. Perceptual-hash grouping ([DL-GROUP])
# --------------------------------------------------------------------------- #
def _phash_bits(path: str, hash_size: int) -> Optional[np.ndarray]:
    import imagehash
    try:
        with Image.open(path) as im:
            h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        return h.hash.flatten()  # boolean array
    except Exception as e:  # corrupt image -> own group
        log.warning("phash failed for %s (%s)", path, e)
        return None


def assign_groups(cfg: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'group' column. Near-duplicates (Hamming <= thr) share a group so
    they never straddle train/test. Without this, the same fruit/scene leaks
    across splits and inflates test accuracy — the exact artifact this project
    is trying to detect.
    """
    gcfg = cfg["split"]["group"]
    df = df.reset_index(drop=True).copy()
    if not gcfg["use_perceptual_hash"]:
        df["group"] = np.arange(len(df))  # each image its own group
        log.info("Group split disabled: every image is its own group.")
        return df

    n = len(df)
    bits = np.zeros((n, cfg["split"]["group"]["phash_size"] ** 2), dtype=bool)
    valid = np.ones(n, dtype=bool)
    for i, p in enumerate(df["path"].tolist()):
        b = _phash_bits(p, gcfg["phash_size"])
        if b is None:
            valid[i] = False
        else:
            bits[i] = b

    thr = gcfg["near_dup_hamming"]
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # O(n^2) Hamming, vectorized per-row (fine for a few-thousand-image dataset).
    for i in range(n):
        if not valid[i]:
            continue
        xor = np.logical_xor(bits[i + 1:], bits[i])  # (n-i-1, bits)
        dist = xor.sum(axis=1)
        for off in np.where(dist <= thr)[0]:
            j = i + 1 + off
            if valid[j]:
                union(i, j)

    groups = np.array([find(i) for i in range(n)])
    # remap to dense ids
    _, df["group"] = np.unique(groups, return_inverse=True)
    n_groups = df["group"].nunique()
    n_dups = n - n_groups
    log.info("Perceptual hashing: %d images -> %d groups (%d near-duplicates "
             "collapsed). Largest group: %d images.",
             n, n_groups, n_dups, df["group"].value_counts().max())
    return df


# --------------------------------------------------------------------------- #
# 3. Group-aware stratified split ([DL-SPLIT] [DL-OOD])
# --------------------------------------------------------------------------- #
def make_splits(cfg: Dict[str, Any], df: pd.DataFrame,
                seed: int) -> pd.DataFrame:
    method = cfg["split"]["method"]
    test_frac = cfg["split"]["test"]
    val_frac = cfg["split"]["val"]
    df = df.reset_index(drop=True).copy()
    y = df["label_idx"].to_numpy()

    if method == "random":
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=test_frac, random_state=seed)
        tr, va = train_test_split(tr, test_size=val_frac / (1 - test_frac),
                                  random_state=seed)
    elif method == "stratified":
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=test_frac, random_state=seed,
                                  stratify=y)
        tr, va = train_test_split(tr, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=y[tr])
    elif method == "group_stratified":
        tr, va, te = _group_stratified_split(df, y, test_frac, val_frac, seed)
    else:
        raise ValueError(f"unknown split method '{method}'")

    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    _report_split(df, cfg["data"]["classes"])
    return df


def _group_stratified_split(df, y, test_frac, val_frac, seed):
    """No group spans two splits. Approximate per-class stratification using
    sklearn's StratifiedGroupKFold with a fold count matched to the target
    fraction."""
    from sklearn.model_selection import StratifiedGroupKFold
    groups = df["group"].to_numpy()
    idx = np.arange(len(df))

    n_splits_test = max(2, round(1.0 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True,
                                random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(idx, y, groups))

    rem_frac = val_frac / (1.0 - test_frac)
    n_splits_val = max(2, round(1.0 / rem_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True,
                                 random_state=seed)
    sub_train, sub_val = next(
        sgkf2.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
    train_idx = trainval_idx[sub_train]
    val_idx = trainval_idx[sub_val]

    # leakage assertion: groups disjoint across splits
    for a, b in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
        assert not (set(groups[a]) & set(groups[b])), "group leakage detected!"
    return train_idx, val_idx, test_idx


def _report_split(df: pd.DataFrame, classes: List[str]) -> None:
    tab = (df.groupby(["split", "label"]).size()
           .unstack(fill_value=0).reindex(columns=classes, fill_value=0))
    log.info("Split x class counts:\n%s", tab.to_string())
    frac = df["split"].value_counts(normalize=True).round(3).to_dict()
    log.info("Realized split fractions: %s", frac)


def save_splits(df: pd.DataFrame, path: str) -> None:
    df = df.copy()
    if "group" not in df.columns:          # predefined split w/o leakage audit
        df["group"] = np.arange(len(df))
    save_json({"records": df[["path", "label", "label_idx", "group", "split"]]
               .to_dict(orient="records")}, path)
    log.info("Saved splits -> %s", path)


def load_splits(path: str) -> pd.DataFrame:
    import json
    with open(path) as f:
        return pd.DataFrame(json.load(f)["records"])


# --------------------------------------------------------------------------- #
# 4. Transforms ([DL-RESIZE] [DL-NORM] [DL-AUG-*])
# --------------------------------------------------------------------------- #
def _interp(name: str):
    from torchvision.transforms import InterpolationMode
    return {"bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC}[name]


def _norm_stats(cfg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    n = cfg["normalization"]
    if n["use_imagenet_stats"] or n["dataset_mean"] is None:
        return n["imagenet_mean"], n["imagenet_std"]
    return n["dataset_mean"], n["dataset_std"]


def build_transforms(cfg: Dict[str, Any], train: bool):
    import torchvision.transforms as T
    size = cfg["image"]["size"]
    interp = _interp(cfg["image"]["interpolation"])
    mean, std = _norm_stats(cfg)

    if not train:
        return T.Compose([
            T.Resize(cfg["image"]["resize_shorter"], interpolation=interp),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    a = cfg["augment"]
    ops: List[Any] = []
    if a["random_resized_crop"]["enabled"]:
        ops.append(T.RandomResizedCrop(
            size, scale=tuple(a["random_resized_crop"]["scale"]),
            ratio=tuple(a["random_resized_crop"]["ratio"]), interpolation=interp))
    else:
        ops += [T.Resize(cfg["image"]["resize_shorter"], interpolation=interp),
                T.CenterCrop(size)]
    if a["horizontal_flip"] > 0:
        ops.append(T.RandomHorizontalFlip(p=a["horizontal_flip"]))
    if a["rotation_degrees"] > 0:
        ops.append(T.RandomRotation(a["rotation_degrees"], interpolation=interp))
    cj = a["color_jitter"]
    if cj["enabled"] and any([cj["brightness"], cj["contrast"],
                              cj["saturation"], cj["hue"]]):
        # saturation/hue default to 0 in config -> colour cue preserved.
        ops.append(T.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"],
                                 saturation=cj["saturation"], hue=cj["hue"]))
    if a["randaugment"]["enabled"]:
        ops.append(T.RandAugment(num_ops=a["randaugment"]["num_ops"],
                                 magnitude=a["randaugment"]["magnitude"]))
    elif a["trivialaugment"]["enabled"]:
        ops.append(T.TrivialAugmentWide())
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


# --------------------------------------------------------------------------- #
# 5. Dataset + dataloaders
# --------------------------------------------------------------------------- #
class DurianDataset:
    """Minimal torch Dataset (imported lazily so EDA can run without torch)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[i]


def build_dataloaders(cfg: Dict[str, Any], df: pd.DataFrame, seed: int,
                      smoke: bool = False):
    import torch
    from torch.utils.data import DataLoader

    bs = cfg["smoke_test"]["batch_size"] if smoke else cfg["train"]["batch_size"]
    nw = 0 if smoke else cfg["train"]["num_workers"]

    if smoke:  # subsample per class for a fast CPU smoke test
        k = cfg["smoke_test"]["subset_per_class"]
        df = (df.groupby("label", group_keys=False)
              .apply(lambda g: g.head(k)).reset_index(drop=True))

    loaders = {}
    g = torch.Generator()
    g.manual_seed(seed)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        ds = DurianDataset(sub, build_transforms(cfg, train=(split == "train")))
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"),
            num_workers=nw, pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker, generator=g, drop_last=False)
    return loaders


def build_ood_loader(cfg: Dict[str, Any], seed: int):
    """Quarantined OOD set ([DL-OOD]): eval only, never seen in train/select."""
    import torch
    from torch.utils.data import DataLoader
    df = discover_images(cfg, data_root=cfg["paths"]["ood_root"])
    df["split"] = "ood"
    ds = DurianDataset(df, build_transforms(cfg, train=False))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"],
                        pin_memory=torch.cuda.is_available())
    return loader, df
