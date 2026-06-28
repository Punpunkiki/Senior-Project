"""
utils.py — reproducibility, config loading, device, logging, small helpers.

Design note: ALL numeric hyper-parameters come from config.yaml. This module
only contains plumbing (seeding, IO, metrics wrappers), never tunable numbers.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:  # torch is optional for pure-EDA usage of this module
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load the single-source-of-truth YAML config."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def model_cfg_by_name(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return the per-model config block for `name` (raises if absent)."""
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    valid = [m["name"] for m in cfg["models"]]
    raise KeyError(f"model '{name}' not in config; valid: {valid}")


# --------------------------------------------------------------------------- #
# Reproducibility  ([DL-PRETRAIN] reproducibility-first requirement)
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch / cuda. Document residual non-determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn.deterministic trades a little speed for reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Some ops (e.g. certain pooling/interp) remain non-deterministic on
            # GPU; we allow them rather than crash, and note it in the README.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:  # for DataLoader worker_init_fn
    worker_seed = (torch.initial_seed() % 2 ** 32) if _HAS_TORCH else worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> "torch.device":
    if not _HAS_TORCH:
        raise RuntimeError("torch not installed")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# IO / logging
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def get_logger(name: str = "durian", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------- #
# Small training helpers
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_params(model) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class LatencyResult:
    mean_ms: float
    std_ms: float
    batch_size: int
    device: str


def measure_latency(model, input_size: int, device, batch_size: int = 1,
                    warmup: int = 5, iters: int = 30) -> LatencyResult:
    """Measure single-sample (or batch) forward latency. Deployment metric."""
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    times: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return LatencyResult(float(arr.mean()), float(arr.std()), batch_size, device.type)
