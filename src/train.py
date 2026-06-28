"""
train.py — reproducible, seeded, mixed-precision training for one model.

Run:
  python -m src.train --model convnextv2_tiny
  python -m src.train --model swin_tiny --smoke         # fast CPU sanity run
  python -m src.train --all                             # train all 3 in turn

Decision Logs implemented here: [DL-PRETRAIN] [DL-FINETUNE] [DL-LOSS]
[DL-OPT] [DL-STOP].  Curves + best weights + history -> outputs/<model>/.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data import (assign_groups, build_dataloaders, discover_images,
                   load_splits, make_splits, save_splits)
from .models import (build_model, build_param_groups, freeze_backbone,
                     unfreeze_all)
from .utils import (AverageMeter, count_params, ensure_dir, get_device,
                    get_logger, load_config, save_json, set_seed)

log = get_logger("train")


# --------------------------------------------------------------------------- #
# Loss / optim / schedule
# --------------------------------------------------------------------------- #
def build_loss(cfg: Dict[str, Any], class_weights=None):
    """[DL-LOSS] CE+label-smoothing default; focal only argued for imbalance."""
    import torch
    import torch.nn as nn
    lcfg = cfg["train"]["loss"]
    w = None
    if class_weights is not None:
        w = torch.as_tensor(class_weights, dtype=torch.float32)
    if lcfg["name"] == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if lcfg["name"] == "ce_label_smoothing":
        return nn.CrossEntropyLoss(weight=w,
                                   label_smoothing=lcfg["label_smoothing"])
    if lcfg["name"] == "focal":
        return _FocalLoss(gamma=lcfg["focal_gamma"], weight=w)
    raise ValueError(f"unknown loss '{lcfg['name']}'")


class _FocalLoss:
    def __init__(self, gamma: float, weight=None):
        import torch.nn as nn
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def __call__(self, logits, target):
        import torch
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_optimizer(cfg: Dict[str, Any], model, name: str):
    import torch
    groups = build_param_groups(cfg, model, name)
    o = cfg["train"]["optimizer"]
    if o["name"] == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if o["name"] == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer '{o['name']}'")


def build_scheduler(cfg: Dict[str, Any], optimizer, total_epochs: int):
    """[DL-OPT] linear warmup -> cosine decay (warmup is critical for Swin)."""
    import torch
    s = cfg["train"]["scheduler"]
    warmup = s["warmup_epochs"]
    o = cfg["train"]["optimizer"]
    ref_lr = max(o["lr"], o["backbone_lr"])
    floor = s["min_lr"] / ref_lr if ref_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        prog = (epoch - warmup) / float(max(1, total_epochs - warmup))
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return floor + (1.0 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Data prep (shared with evaluate.py)
# --------------------------------------------------------------------------- #
def prepare_dataframe(cfg: Dict[str, Any], seed: int):
    """Build (or reuse) the split dataframe so train/eval see identical splits."""
    splits_file = cfg["paths"]["splits_file"]
    if Path(splits_file).exists():
        log.info("Reusing existing splits: %s", splits_file)
        return load_splits(splits_file)
    df = discover_images(cfg)
    df = assign_groups(cfg, df)
    df = make_splits(cfg, df, seed)
    save_splits(df, splits_file)
    return df


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, criterion, optimizer, scaler, device,
               train: bool, grad_clip: float, mixup_fn=None):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    model.train(train)
    loss_meter = AverageMeter()
    ys, ps = [], []
    use_amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_eval = y.clone()
        if train and mixup_fn is not None:
            x, y = mixup_fn(x, y)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        loss_meter.update(loss.item(), x.size(0))
        ys.append(y_eval.cpu().numpy())
        ps.append(logits.detach().argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {"loss": loss_meter.avg,
            "acc": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


# --------------------------------------------------------------------------- #
# Train one model
# --------------------------------------------------------------------------- #
def train_one(cfg: Dict[str, Any], name: str, df, device, smoke: bool = False,
              pretrained: Optional[bool] = None) -> Dict[str, Any]:
    import torch
    from timm.data import Mixup

    seed = cfg["seed"]
    out_dir = ensure_dir(Path(cfg["paths"]["outputs_dir"]) / name)
    epochs = cfg["smoke_test"]["epochs"] if smoke else cfg["train"]["epochs"]
    pre = cfg["train"]["pretrained"] if pretrained is None else pretrained

    loaders = build_dataloaders(cfg, df, seed, smoke=smoke)
    model = build_model(cfg, name, cfg["data"]["num_classes"], pretrained=pre).to(device)

    # --- fine-tuning regime ([DL-FINETUNE]) ---
    strat = cfg["train"]["finetune"]["strategy"]
    warmup_lp = cfg["train"]["finetune"]["linear_probe_warmup_epochs"]
    if strat == "linear_probe":
        freeze_backbone(model)
    elif strat in ("gradual",) and warmup_lp > 0:
        freeze_backbone(model)  # body unfrozen after warmup_lp epochs below

    optimizer = build_optimizer(cfg, model, name)
    scheduler = build_scheduler(cfg, optimizer, epochs)
    criterion = build_loss(cfg, cfg["train"]["loss"]["class_weights"])

    # --- optional Mixup/CutMix ([DL-AUG-OFF] by default) ---
    mixup_fn = None
    acfg = cfg["augment"]
    if acfg["mixup"]["enabled"] or acfg["cutmix"]["enabled"]:
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(
            mixup_alpha=acfg["mixup"]["alpha"] if acfg["mixup"]["enabled"] else 0.0,
            cutmix_alpha=acfg["cutmix"]["alpha"] if acfg["cutmix"]["enabled"] else 0.0,
            label_smoothing=cfg["train"]["loss"]["label_smoothing"],
            num_classes=cfg["data"]["num_classes"])
        criterion = SoftTargetCrossEntropy()
        log.info("Mixup/CutMix enabled -> SoftTargetCrossEntropy.")

    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["amp"] and
                                             device.type == "cuda") else None
    total, trainable = count_params(model)
    log.info("%s params: total=%.2fM trainable=%.2fM", name, total / 1e6,
             trainable / 1e6)

    es = cfg["train"]["early_stopping"]
    best, best_epoch, patience = -math.inf, -1, 0
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        if strat == "gradual" and epoch == warmup_lp:
            unfreeze_all(model)
            optimizer = build_optimizer(cfg, model, name)  # refresh groups
            scheduler = build_scheduler(cfg, optimizer, epochs)
            log.info("[gradual] unfroze backbone at epoch %d", epoch)

        tr = _run_epoch(model, loaders["train"], criterion, optimizer, scaler,
                        device, True, cfg["train"]["grad_clip_norm"], mixup_fn)
        va = _run_epoch(model, loaders["val"], build_loss(cfg), None, None,
                        device, False, 0.0, None)
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]["lr"]
        rec = {"epoch": epoch, "lr": cur_lr,
               "train_loss": tr["loss"], "train_acc": tr["acc"],
               "train_macro_f1": tr["macro_f1"], "val_loss": va["loss"],
               "val_acc": va["acc"], "val_macro_f1": va["macro_f1"]}
        history.append(rec)
        log.info("[%s] ep %02d/%d | train f1 %.3f | val f1 %.3f acc %.3f | lr %.2g",
                 name, epoch, epochs - 1, tr["macro_f1"], va["macro_f1"],
                 va["acc"], cur_lr)

        monitor = va[es["monitor"].replace("val_", "")]
        if monitor > best + es["min_delta"]:
            best, best_epoch, patience = monitor, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best, "config_model": name},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= es["patience"]:
                log.info("[%s] early stop at epoch %d (best %s=%.3f @ ep %d)",
                         name, epoch, es["monitor"], best, best_epoch)
                break

    save_json({"model": name, "best_val_macro_f1": best,
               "best_epoch": best_epoch, "epochs_run": len(history),
               "params_total": total, "params_trainable": trainable,
               "history": history}, out_dir / "history.json")
    _plot_curves(history, out_dir / "curves.png", name)
    log.info("[%s] DONE. best val macro-F1=%.4f @ epoch %d -> %s",
             name, best, best_epoch, out_dir / "best.pt")
    return {"model": name, "best_val_macro_f1": best, "best_epoch": best_epoch}


def _plot_curves(history, path, name) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title(f"{name}: loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_macro_f1"] for h in history], label="train")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val")
    ax[1].set_title(f"{name}: macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train durian disease classifier")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="single model name to train")
    ap.add_argument("--all", action="store_true", help="train all 3 models")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke run")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init (code-path validation only; not for metrics)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    log.info("Device: %s | seed: %d", device, cfg["seed"])

    df = prepare_dataframe(cfg, cfg["seed"])
    pre = False if args.no_pretrained else None

    names = [m["name"] for m in cfg["models"]] if args.all else \
        [args.model or cfg["models"][0]["name"]]
    results = [train_one(cfg, n, df, device, smoke=args.smoke, pretrained=pre)
               for n in names]
    log.info("Summary: %s", results)


if __name__ == "__main__":
    main()
