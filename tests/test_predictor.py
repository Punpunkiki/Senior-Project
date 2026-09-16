"""Unit tests for app/ml/predictor.py.

Uses the real config.yaml (classes/architectures) but points outputs_dir at
a tmp directory, since the repo genuinely has no trained checkpoint yet --
that "not ready" path is the real current state and must be tested first.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from app.ml.predictor import ClassScore, DurianPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "efficientnet_b0"  # smallest of the 3 -> fastest test


def _write_cfg(tmp_path: Path, outputs_dir: Path) -> Path:
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["outputs_dir"] = str(outputs_dir)
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (300, 300), color=(120, 180, 90))


def test_not_ready_without_checkpoint(tmp_path):
    outputs_dir = tmp_path / "outputs"
    cfg_path = _write_cfg(tmp_path, outputs_dir)

    predictor = DurianPredictor(cfg_path, MODEL_NAME)

    assert predictor.ready is False
    with pytest.raises(RuntimeError, match="not ready"):
        predictor.predict(_dummy_image())


def test_ready_and_predicts_top_k_after_checkpoint(tmp_path):
    outputs_dir = tmp_path / "outputs"
    cfg_path = _write_cfg(tmp_path, outputs_dir)

    # Build a checkpoint the same way src/train.py does, but with a random
    # (untrained) backbone -- this test only exercises the serving plumbing,
    # never claims real accuracy.
    from src.models import build_model
    from src.utils import load_config

    cfg = load_config(cfg_path)
    model = build_model(cfg, MODEL_NAME, cfg["data"]["num_classes"], pretrained=False)
    model_dir = outputs_dir / MODEL_NAME
    model_dir.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "epoch": 1, "val_macro_f1": 0.42},
               model_dir / "best.pt")

    predictor = DurianPredictor(cfg_path, MODEL_NAME)

    assert predictor.ready is True
    assert predictor.checkpoint_val_macro_f1 == pytest.approx(0.42)
    assert predictor.temperature == 1.0  # no metrics.json -> uncalibrated default

    top3 = predictor.predict(_dummy_image(), top_k=3)

    assert len(top3) == 3
    assert all(isinstance(s, ClassScore) for s in top3)
    assert all(s.label in cfg["data"]["classes"] for s in top3)
    assert all(0.0 <= s.confidence <= 1.0 for s in top3)
    confidences = [s.confidence for s in top3]
    assert confidences == sorted(confidences, reverse=True)


def test_temperature_loaded_from_metrics_json(tmp_path):
    outputs_dir = tmp_path / "outputs"
    cfg_path = _write_cfg(tmp_path, outputs_dir)

    from src.models import build_model
    from src.utils import load_config, save_json

    cfg = load_config(cfg_path)
    model = build_model(cfg, MODEL_NAME, cfg["data"]["num_classes"], pretrained=False)
    model_dir = outputs_dir / MODEL_NAME
    model_dir.mkdir(parents=True)
    torch.save({"model": model.state_dict()}, model_dir / "best.pt")
    save_json({"temperature": 2.37}, model_dir / "metrics.json")

    predictor = DurianPredictor(cfg_path, MODEL_NAME)

    assert predictor.temperature == pytest.approx(2.37)
