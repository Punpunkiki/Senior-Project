"""
evaluate.py — metrics, confusion matrix, calibration, temperature scaling,
and confidence-threshold / abstain selection.  ([DL-METRICS] [DL-THRESH])

Run:
  python -m src.evaluate --model convnextv2_tiny            # in-distribution test
  python -m src.evaluate --model convnextv2_tiny --ood      # + OOD set (Phase 9)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data import build_dataloaders, build_ood_loader
from .models import build_model
from .train import prepare_dataframe
from .utils import (ensure_dir, get_device, get_logger, load_config, save_json,
                    set_seed)

log = get_logger("evaluate")


# --------------------------------------------------------------------------- #
# Prediction collection
# --------------------------------------------------------------------------- #
def collect_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits_all.append(model(x).float().cpu().numpy())
            y_all.append(np.asarray(y))
    return np.concatenate(logits_all), np.concatenate(y_all)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Core metrics ([DL-METRICS])
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, classes: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, classes: List[str], path,
                          title: str) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return cm


# --------------------------------------------------------------------------- #
# Calibration ([DL-METRICS]: why calibration matters before thresholding)
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs, y_true, n_bins: int) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_diagram(probs, y_true, n_bins: int, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        centers.append((lo + hi) / 2); accs.append(correct[m].mean())
        confs.append(conf[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(confs, accs, "o-", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(title); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Temperature scaling ([DL-THRESH]: calibrate-then-threshold)
# --------------------------------------------------------------------------- #
def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Optimise a single scalar T (>0) minimising NLL on the val set."""
    import torch
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) keeps T>0
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_T), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).item())
    log.info("Fitted temperature T=%.3f", T)
    return T


# --------------------------------------------------------------------------- #
# Confidence threshold / abstain ([DL-THRESH])
# --------------------------------------------------------------------------- #
def precision_coverage_curve(probs, y_true, cfg) -> Dict[str, np.ndarray]:
    taus = np.linspace(cfg["threshold"]["sweep_start"],
                       cfg["threshold"]["sweep_stop"],
                       cfg["threshold"]["sweep_steps"])
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true)
    cov, prec = [], []
    for t in taus:
        keep = conf >= t
        cov.append(keep.mean())
        prec.append(correct[keep].mean() if keep.any() else 1.0)
    return {"tau": taus, "coverage": np.array(cov), "precision": np.array(prec)}


def choose_threshold(probs, y_true, cfg,
                     ood_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Pick the abstain threshold. Default: smallest tau hitting target precision
    on the in-distribution val set (precision/coverage trade-off). Alternative:
    best ID-vs-OOD separation when an OOD sample is supplied."""
    method = cfg["threshold"]["method"]
    curve = precision_coverage_curve(probs, y_true, cfg)
    target = cfg["threshold"]["target_precision"]

    if method == "id_ood_separation" and ood_probs is not None:
        taus = curve["tau"]
        id_conf = probs.max(axis=1)
        ood_conf = ood_probs.max(axis=1)
        # treat "keep as in-distribution" as positive; maximise Youden's J
        j = []
        for t in taus:
            tpr = (id_conf >= t).mean()         # ID correctly kept
            fpr = (ood_conf >= t).mean()        # OOD wrongly kept
            j.append(tpr - fpr)
        best = int(np.argmax(j))
        chosen = float(taus[best])
        rationale = f"id_ood_separation: max Youden's J={j[best]:.3f}"
    else:  # precision_coverage
        ok = np.where(curve["precision"] >= target)[0]
        if len(ok):
            chosen = float(curve["tau"][ok[0]])
            rationale = (f"precision_coverage: smallest tau with "
                         f"precision>={target}")
        else:
            chosen = float(curve["tau"][np.argmax(curve["precision"])])
            rationale = (f"precision_coverage: target {target} unreachable; "
                         f"chose max-precision tau")
    cov_at = float((probs.max(axis=1) >= chosen).mean())
    return {"threshold": chosen, "rationale": rationale,
            "coverage_at_threshold": cov_at,
            "curve": {k: v.tolist() for k, v in curve.items()}}


def plot_threshold_tradeoff(curve: Dict[str, np.ndarray], chosen: float,
                            path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["tau"], curve["coverage"], label="coverage")
    ax.plot(curve["tau"], curve["precision"], label="precision (selective acc)")
    ax.axvline(chosen, color="r", ls="--", label=f"chosen tau={chosen:.2f}")
    ax.set_xlabel("softmax confidence threshold tau"); ax.set_ylabel("rate")
    ax.legend(); ax.set_title("Precision / coverage trade-off")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate_model(cfg, name, device, df, with_ood: bool = False,
                   pretrained_fallback: bool = False) -> Dict[str, Any]:
    import torch
    classes = cfg["data"]["classes"]
    out_dir = ensure_dir(Path(cfg["paths"]["outputs_dir"]) / name)
    ckpt = out_dir / "best.pt"
    model = build_model(cfg, name, cfg["data"]["num_classes"],
                        pretrained=pretrained_fallback).to(device)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        log.info("Loaded weights %s (val_macro_f1=%.4f)", ckpt,
                 state.get("val_macro_f1", float("nan")))
    else:
        log.warning("No checkpoint at %s; evaluating an UNTRAINED model. "
                    "Metrics below are NOT valid — train first.", ckpt)

    loaders = build_dataloaders(cfg, df, cfg["seed"])
    val_logits, val_y = collect_logits(model, loaders["val"], device)
    test_logits, test_y = collect_logits(model, loaders["test"], device)

    # temperature scaling fit on val ([DL-THRESH])
    T = fit_temperature(val_logits, val_y) if cfg["threshold"]["temperature_scaling"] else 1.0
    val_probs = softmax(val_logits / T)
    test_probs = softmax(test_logits / T)
    test_pred = test_probs.argmax(1)

    metrics = compute_metrics(test_y, test_pred, classes)
    metrics["temperature"] = T
    metrics["ece_raw"] = expected_calibration_error(
        softmax(test_logits), test_y, cfg["eval"]["calibration_bins"])
    metrics["ece_scaled"] = expected_calibration_error(
        test_probs, test_y, cfg["eval"]["calibration_bins"])

    plot_confusion_matrix(test_y, test_pred, classes,
                          out_dir / "confusion_matrix.png",
                          f"{name}: in-distribution test")
    reliability_diagram(test_probs, test_y, cfg["eval"]["calibration_bins"],
                        out_dir / "reliability.png", f"{name}: reliability (T={T:.2f})")

    # threshold chosen on VAL ([DL-THRESH]); optionally use OOD separation
    ood_probs = None
    if with_ood:
        try:
            ood_loader, ood_df = build_ood_loader(cfg, cfg["seed"])
            ood_logits, ood_y = collect_logits(model, ood_loader, device)
            ood_probs = softmax(ood_logits / T)
            ood_pred = ood_probs.argmax(1)
            metrics["ood"] = compute_metrics(ood_y, ood_pred, classes)
            plot_confusion_matrix(ood_y, ood_pred, classes,
                                  out_dir / "confusion_matrix_ood.png",
                                  f"{name}: OOD (web/social)")
            metrics["ood"]["id_minus_ood_macro_f1"] = (
                metrics["macro_f1"] - metrics["ood"]["macro_f1"])
        except FileNotFoundError as e:
            log.warning("OOD eval skipped: %s", e)

    thr = choose_threshold(val_probs, val_y, cfg, ood_probs=ood_probs)
    plot_threshold_tradeoff({k: np.array(v) for k, v in thr["curve"].items()},
                            thr["threshold"], out_dir / "threshold_tradeoff.png")
    # apply abstain on test
    kept = test_probs.max(1) >= thr["threshold"]
    metrics["abstain"] = {
        "threshold": thr["threshold"], "rationale": thr["rationale"],
        "test_coverage": float(kept.mean()),
        "test_selective_accuracy": float((test_pred[kept] == test_y[kept]).mean()
                                         if kept.any() else float("nan"))}
    thr.pop("curve", None)
    save_json(metrics, out_dir / "metrics.json")
    log.info("[%s] test acc=%.4f macro-F1=%.4f ECE(raw->scaled)=%.3f->%.3f",
             name, metrics["accuracy"], metrics["macro_f1"],
             metrics["ece_raw"], metrics["ece_scaled"])
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate durian classifier")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ood", action="store_true", help="also evaluate OOD set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    df = prepare_dataframe(cfg, cfg["seed"])
    names = [m["name"] for m in cfg["models"]] if args.all else \
        [args.model or cfg["models"][0]["name"]]
    for n in names:
        evaluate_model(cfg, n, device, df, with_ood=args.ood)


if __name__ == "__main__":
    main()
