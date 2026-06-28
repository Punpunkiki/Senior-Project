"""
eda.py — Phase 1 exploratory analysis as reusable functions (so the notebook
and a headless run share identical, no-magic-number logic).

Produces, into outputs/eda/:
  class_counts.png / .csv      verified per-class counts (the TRUTH)
  resolution_scatter.png       image size / aspect-ratio distribution
  brightness_by_class.png      mean brightness + per-channel means per class
  dataset_norm_stats.json      dataset-computed mean/std (informational)
  duplicates.json              near-duplicate clusters via perceptual hashing
  eda_summary.json             machine-readable digest used by the report

Run headless:  python -m src.eda
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

from .data import assign_groups, discover_images
from .utils import ensure_dir, get_logger, load_config, save_json

log = get_logger("eda")


def class_counts(df: pd.DataFrame, classes, out_dir: Path) -> Dict[str, int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = df["label"].value_counts().reindex(classes).fillna(0).astype(int)
    counts.to_csv(out_dir / "class_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot.bar(ax=ax)
    ax.axhline(counts.mean(), color="r", ls="--",
               label=f"mean={counts.mean():.0f}")
    ax.set_title("Verified per-class image counts"); ax.legend()
    plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(out_dir / "class_counts.png", dpi=120); plt.close(fig)
    return counts.to_dict()


def resolution_stats(df: pd.DataFrame, out_dir: Path, sample: int = 1500) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df.sample(min(sample, len(df)), random_state=0)
    ws, hs = [], []
    for p in sub["path"]:
        try:
            with Image.open(p) as im:
                ws.append(im.width); hs.append(im.height)
        except Exception:
            continue
    ws, hs = np.array(ws), np.array(hs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ws, hs, s=5, alpha=0.3)
    ax.set_xlabel("width"); ax.set_ylabel("height")
    ax.set_title("Image resolution distribution (sample)")
    fig.tight_layout(); fig.savefig(out_dir / "resolution_scatter.png", dpi=120)
    plt.close(fig)
    ar = ws / np.clip(hs, 1, None)
    return {"width_median": int(np.median(ws)), "height_median": int(np.median(hs)),
            "aspect_ratio_median": float(np.median(ar)),
            "min_side_p05": int(np.percentile(np.minimum(ws, hs), 5))}


def brightness_and_norm(df: pd.DataFrame, classes, out_dir: Path,
                        sample_per_class: int = 120) -> Dict[str, Any]:
    """Per-class brightness/channel means (informs colour-aug safety) AND
    dataset-wide mean/std ([DL-NORM])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    chan_sum = np.zeros(3); chan_sqsum = np.zeros(3); n_pix = 0
    for cls, g in df.groupby("label"):
        for p in g.sample(min(sample_per_class, len(g)), random_state=0)["path"]:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            except Exception:
                continue
            rows.append({"label": cls, "brightness": arr.mean(),
                         "R": arr[..., 0].mean(), "G": arr[..., 1].mean(),
                         "B": arr[..., 2].mean()})
            chan_sum += arr.reshape(-1, 3).sum(0)
            chan_sqsum += (arr.reshape(-1, 3) ** 2).sum(0)
            n_pix += arr.shape[0] * arr.shape[1]
    bdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    bdf.groupby("label")["brightness"].mean().reindex(classes).plot.bar(ax=ax)
    ax.set_title("Mean brightness per class (colour-aug safety check)")
    plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(out_dir / "brightness_by_class.png", dpi=120); plt.close(fig)

    mean = (chan_sum / max(n_pix, 1)).tolist()
    std = np.sqrt(np.maximum(chan_sqsum / max(n_pix, 1) - np.array(mean) ** 2, 1e-8)).tolist()
    save_json({"dataset_mean": mean, "dataset_std": std,
               "imagenet_mean": [0.485, 0.456, 0.406],
               "imagenet_std": [0.229, 0.224, 0.225],
               "note": "Default uses ImageNet stats for transfer ([DL-NORM])."},
              out_dir / "dataset_norm_stats.json")
    return {"dataset_mean": mean, "dataset_std": std,
            "channel_means_by_class": bdf.groupby("label")[["R", "G", "B"]].mean()
            .reindex(classes).round(3).to_dict(orient="index")}


def duplicate_report(cfg, df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    grouped = assign_groups(cfg, df)
    sizes = grouped["group"].value_counts()
    dup_groups = sizes[sizes > 1]
    # leakage flag: groups whose members span >1 class (background-driven dupes)
    cross = (grouped.groupby("group")["label"].nunique())
    cross_class = cross[cross > 1]
    report = {
        "n_images": int(len(grouped)),
        "n_groups": int(grouped["group"].nunique()),
        "n_near_duplicate_images": int(len(grouped) - grouped["group"].nunique()),
        "n_groups_with_duplicates": int((sizes > 1).sum()),
        "largest_group_size": int(sizes.max()),
        "n_cross_class_groups": int(len(cross_class)),
        "note": ("Cross-class near-duplicate groups (same scene, different label) "
                 "are a strong artifact/leakage signal -> group-aware split "
                 "[DL-GROUP] is mandatory."),
    }
    save_json(report, out_dir / "duplicates.json")
    return report


def run_eda(cfg) -> Dict[str, Any]:
    classes = cfg["data"]["classes"]
    out_dir = ensure_dir(cfg["paths"]["eda_dir"])
    df = discover_images(cfg)
    summary = {
        "image_format": cfg["data"]["image_format"],
        "class_counts": class_counts(df, classes, out_dir),
        "resolution": resolution_stats(df, out_dir),
        "brightness_norm": brightness_and_norm(df, classes, out_dir),
        "duplicates": duplicate_report(cfg, df, out_dir),
    }
    save_json(summary, out_dir / "eda_summary.json")
    log.info("EDA complete -> %s", out_dir)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run_eda(load_config(args.config))


if __name__ == "__main__":
    main()
