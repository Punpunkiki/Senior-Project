"""
interpret.py — Grad-CAM / attention visualisation to answer the project's
CENTRAL QUESTION: does the model attend to lesions, or to background/artifacts?
([DL-INTERPRET])

Outputs:
  outputs/<model>/gradcam/<class>_<correct|wrong>_<i>.png   (overlays)
  outputs/<model>/occlusion_summary.json                    (quantitative proxy)

The occlusion test is a label-free quantitative proxy for "looks at the right
thing": if masking the CAM-highlighted region collapses the prediction far more
than masking a random region, the model is genuinely using that region.

Run:  python -m src.interpret --model convnextv2_tiny
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .data import build_dataloaders, build_transforms
from .models import build_model, gradcam_target_layers, swin_reshape_transform
from .train import prepare_dataframe
from .utils import (ensure_dir, get_device, get_logger, load_config, save_json,
                    set_seed)

log = get_logger("interpret")


def _denorm(t, cfg) -> np.ndarray:
    mean = np.array(cfg["normalization"]["imagenet_mean"]).reshape(3, 1, 1)
    std = np.array(cfg["normalization"]["imagenet_std"]).reshape(3, 1, 1)
    img = t.cpu().numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def _make_cam(model, name, device):
    from pytorch_grad_cam import GradCAM
    targets = gradcam_target_layers(model, name)
    reshape = swin_reshape_transform if name.startswith("swin") else None
    return GradCAM(model=model, target_layers=targets, reshape_transform=reshape)


def run_interpret(cfg, name, device, df, pretrained_fallback=False) -> Dict[str, Any]:
    import torch
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    classes = cfg["data"]["classes"]
    out_dir = ensure_dir(Path(cfg["paths"]["outputs_dir"]) / name / "gradcam")
    model = build_model(cfg, name, cfg["data"]["num_classes"],
                        pretrained=pretrained_fallback).to(device)
    ckpt = Path(cfg["paths"]["outputs_dir"]) / name / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    else:
        log.warning("No checkpoint; Grad-CAM on an untrained model is meaningless.")
    model.eval()

    cam = _make_cam(model, name, device)
    loaders = build_dataloaders(cfg, df, cfg["seed"])
    per_class = cfg["interpret"]["examples_per_class"]
    seen: Dict[int, int] = {c: 0 for c in range(len(classes))}
    occl_drop_cam, occl_drop_rand = [], []

    for x, y in loaders["test"]:
        x = x.to(device)
        with torch.no_grad():
            probs = model(x).softmax(1)
        preds = probs.argmax(1)
        for i in range(x.size(0)):
            cls = int(y[i])
            if seen[cls] >= per_class:
                continue
            correct = int(preds[i]) == cls
            grayscale = cam(input_tensor=x[i:i + 1],
                            targets=[ClassifierOutputTarget(int(preds[i]))])[0]
            rgb = _denorm(x[i], cfg)
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            tag = "correct" if correct else f"wrong_as_{classes[int(preds[i])]}"
            _save_img(overlay, out_dir / f"{classes[cls]}_{tag}_{seen[cls]}.png")
            seen[cls] += 1

            if cfg["interpret"]["occlusion_test"]:
                d_cam, d_rand = _occlusion_drop(
                    model, x[i:i + 1], grayscale, int(preds[i]),
                    cfg["interpret"]["occlusion_patch"], device)
                occl_drop_cam.append(d_cam); occl_drop_rand.append(d_rand)
        if all(seen[c] >= per_class for c in seen):
            break

    summary = {
        "model": name,
        "occlusion_mean_drop_cam_region": float(np.mean(occl_drop_cam)) if occl_drop_cam else None,
        "occlusion_mean_drop_random_region": float(np.mean(occl_drop_rand)) if occl_drop_rand else None,
        "interpretation": ("If drop_cam >> drop_random, the model relies on the "
                           "highlighted (ideally lesion) region, not background."),
        "n_examples": int(sum(seen.values())),
    }
    if occl_drop_cam:
        summary["cam_over_random_ratio"] = (
            summary["occlusion_mean_drop_cam_region"] /
            max(summary["occlusion_mean_drop_random_region"], 1e-6))
    save_json(summary, Path(cfg["paths"]["outputs_dir"]) / name / "occlusion_summary.json")
    log.info("[%s] Grad-CAM saved (%d images). Occlusion summary: %s",
             name, sum(seen.values()), summary.get("cam_over_random_ratio"))
    return summary


def _occlusion_drop(model, x, grayscale, cls, patch, device) -> tuple:
    """Drop in predicted-class prob when occluding the CAM-peak patch vs a random
    patch. Larger CAM drop => model genuinely uses the highlighted region."""
    import torch
    H, W = x.shape[-2:]
    with torch.no_grad():
        base = model(x).softmax(1)[0, cls].item()

    py, px = np.unravel_index(np.argmax(grayscale), grayscale.shape)
    x_cam = _occlude(x.clone(), py, px, patch, H, W)
    ry = np.random.randint(0, max(1, H - patch))
    rx = np.random.randint(0, max(1, W - patch))
    x_rand = _occlude(x.clone(), ry + patch // 2, rx + patch // 2, patch, H, W)
    with torch.no_grad():
        p_cam = model(x_cam).softmax(1)[0, cls].item()
        p_rand = model(x_rand).softmax(1)[0, cls].item()
    return base - p_cam, base - p_rand


def _occlude(x, cy, cx, patch, H, W):
    y0, y1 = max(0, cy - patch // 2), min(H, cy + patch // 2)
    x0, x1 = max(0, cx - patch // 2), min(W, cx + patch // 2)
    x[..., y0:y1, x0:x1] = 0.0  # mean-normalised image -> 0 is the dataset mean
    return x


def _save_img(arr_uint8, path) -> None:
    from PIL import Image
    Image.fromarray(arr_uint8).save(path)


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM interpretability")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    df = prepare_dataframe(cfg, cfg["seed"])
    names = [m["name"] for m in cfg["models"]] if args.all else \
        [args.model or cfg["models"][0]["name"]]
    for n in names:
        run_interpret(cfg, n, device, df)


if __name__ == "__main__":
    main()
