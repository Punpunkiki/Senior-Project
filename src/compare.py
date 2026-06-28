"""
compare.py — single comparison table + plots across the 3 models on
deployment-relevant axes: ID macro-F1, OOD macro-F1, ECE, params, latency.
([DL-FINAL])

Reads outputs/<model>/metrics.json (from evaluate.py --ood) and measures
parameter count + CPU/GPU inference latency live. Writes:
  outputs/comparison.csv
  outputs/comparison.png
  reports/RESULTS_table.md   (paste-ready table for the report)

Run:  python -m src.compare --all
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .models import build_model
from .utils import (count_params, ensure_dir, get_device, get_logger,
                    load_config, load_json, measure_latency)

log = get_logger("compare")


def _latency_and_params(cfg, name, device) -> Dict[str, float]:
    mcfg = next(m for m in cfg["models"] if m["name"] == name)
    model = build_model(cfg, name, cfg["data"]["num_classes"],
                        pretrained=False).to(device)
    total, _ = count_params(model)
    lat = measure_latency(model, mcfg["input_size"], device, batch_size=1)
    return {"params_M": total / 1e6, "latency_ms": lat.mean_ms,
            "latency_device": lat.device}


def build_table(cfg, device, names: List[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        out_dir = Path(cfg["paths"]["outputs_dir"]) / name
        m = load_json(out_dir / "metrics.json") if (out_dir / "metrics.json").exists() else {}
        hw = _latency_and_params(cfg, name, device)
        row = {
            "model": name,
            "family": next(mm["family"] for mm in cfg["models"] if mm["name"] == name),
            "params_M": round(hw["params_M"], 2),
            "id_accuracy": _g(m, "accuracy"),
            "id_macro_f1": _g(m, "macro_f1"),
            "ood_macro_f1": _g(m.get("ood", {}), "macro_f1"),
            "id_minus_ood_f1": _g(m.get("ood", {}), "id_minus_ood_macro_f1"),
            "ece_scaled": _g(m, "ece_scaled"),
            "abstain_threshold": _g(m.get("abstain", {}), "threshold"),
            f"latency_ms_{hw['latency_device']}": round(hw["latency_ms"], 2),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _g(d: Dict[str, Any], k: str):
    v = d.get(k) if isinstance(d, dict) else None
    return round(v, 4) if isinstance(v, (int, float)) else None


def plot_comparison(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(df))
    w = 0.35
    if df["id_macro_f1"].notna().any():
        ax[0].bar(x - w / 2, df["id_macro_f1"].fillna(0), w, label="ID macro-F1")
    if df["ood_macro_f1"].notna().any():
        ax[0].bar(x + w / 2, df["ood_macro_f1"].fillna(0), w, label="OOD macro-F1")
    ax[0].set_xticks(x); ax[0].set_xticklabels(df["model"], rotation=20)
    ax[0].set_title("Generalisation: ID vs OOD macro-F1"); ax[0].legend()

    lat_col = [c for c in df.columns if c.startswith("latency_ms")][0]
    ax[1].scatter(df["params_M"], df["id_macro_f1"].fillna(0), s=80)
    for _, r in df.iterrows():
        ax[1].annotate(r["model"], (r["params_M"], r["id_macro_f1"] or 0),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("params (M)"); ax[1].set_ylabel("ID macro-F1")
    ax[1].set_title("Accuracy vs model size (edge cost)")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description="Cross-model comparison")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    names = args.models or [m["name"] for m in cfg["models"]]

    df = build_table(cfg, device, names)
    out = ensure_dir(cfg["paths"]["outputs_dir"])
    df.to_csv(Path(out) / "comparison.csv", index=False)
    plot_comparison(df, Path(out) / "comparison.png")
    try:
        md = df.to_markdown(index=False)            # needs `tabulate`
    except ImportError:
        md = df.to_string(index=False)
        log.warning("`tabulate` not installed; wrote plain-text table instead.")
    (Path(cfg["paths"]["reports_dir"]) / "RESULTS_table.md").write_text(
        "# Model comparison (auto-generated)\n\n" + md + "\n", encoding="utf-8")
    log.info("Comparison table:\n%s", md)
    print(md)


if __name__ == "__main__":
    main()
