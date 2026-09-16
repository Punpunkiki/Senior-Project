"""app/ml/predictor.py -- wraps the existing training pipeline into a
single-image inference service for the LINE OA bot.

This module never trains or modifies the model. It reuses, unchanged:
  - src/models.py  build_model()      (same timm architecture factory)
  - src/data.py    build_transforms() (same eval-time resize/crop/normalize)
  - outputs/<model_name>/best.pt      (the checkpoint src/train.py writes)
  - outputs/<model_name>/metrics.json (the temperature src/evaluate.py fits)

`DurianPredictor.ready` is False until a real checkpoint is found. Serving
predictions from an untrained backbone would silently fabricate results,
which this project's own decision log forbids -- see reports/RESULTS.md's
"never fabricate numbers" rule. The bot must surface "not ready" instead.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_transforms  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_logger, load_config, load_json  # noqa: E402

log = get_logger("predictor")


@dataclass
class ClassScore:
    label: str
    confidence: float


class DurianPredictor:
    def __init__(self, config_path: str | Path, model_name: str):
        self.cfg = load_config(config_path)
        self.model_name = model_name
        self.classes: List[str] = self.cfg["data"]["classes"]
        self.transform = build_transforms(self.cfg, train=False)
        self.temperature = 1.0
        self.ready = False
        self.checkpoint_val_macro_f1: Optional[float] = None

        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            self.cfg, model_name, self.cfg["data"]["num_classes"], pretrained=False
        ).to(self.device)
        self.model.eval()

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        import torch

        out_dir = Path(self.cfg["paths"]["outputs_dir"]) / self.model_name
        ckpt_path = out_dir / "best.pt"
        if not ckpt_path.exists():
            log.warning(
                "No checkpoint at %s -- predictor NOT ready. Train the model "
                "first (see reports/HANDOFF.md); refusing to serve "
                "predictions from an untrained backbone.",
                ckpt_path,
            )
            return

        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.ready = True
        self.checkpoint_val_macro_f1 = state.get("val_macro_f1")
        log.info(
            "Loaded checkpoint %s (val_macro_f1=%s)",
            ckpt_path,
            self.checkpoint_val_macro_f1,
        )

        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            self.temperature = float(metrics.get("temperature", 1.0))
            log.info(
                "Loaded calibration temperature T=%.3f from %s",
                self.temperature,
                metrics_path,
            )
        else:
            log.warning(
                "No metrics.json at %s -- serving with T=1.0 (uncalibrated). "
                "Run `python -m src.evaluate --model %s` after training.",
                metrics_path,
                self.model_name,
            )

    def predict(self, image: Image.Image, top_k: int = 3) -> List[ClassScore]:
        if not self.ready:
            raise RuntimeError(
                "Predictor not ready: no trained checkpoint loaded for "
                f"'{self.model_name}'."
            )
        import torch

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x) / self.temperature
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        order = probs.argsort()[::-1][:top_k]
        return [ClassScore(self.classes[i], float(probs[i])) for i in order]


_predictor: Optional[DurianPredictor] = None


def get_predictor(config_path: str | Path, model_name: str) -> DurianPredictor:
    """Lazily build the module-level singleton. Call once at FastAPI startup
    (app/main.py) so the model loads a single time, not per-request."""
    global _predictor
    if _predictor is None:
        _predictor = DurianPredictor(config_path, model_name)
    return _predictor


def reset_predictor() -> None:
    """Test-only hook to drop the singleton between test cases."""
    global _predictor
    _predictor = None
