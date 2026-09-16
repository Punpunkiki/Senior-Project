"""app/diagnosis.py -- turns raw top-3 model predictions into a UX-facing
diagnosis tier. Kept separate from app/ml/predictor.py so the confidence-band
decision (a product/UX call) is unit-testable without loading a real model.

Tiering matches the product spec:
  - top-1 class is the reject class ("not_durian")        -> NOT_DURIAN
  - top-1 confidence <  confidence_low  (default 0.45)     -> NOT_CONFIDENT
  - confidence_low <= top-1 confidence < confidence_high    -> UNCERTAIN_TOP2
  - top-1 confidence >= confidence_high (default 0.75)     -> CONFIDENT
`is_healthy` is orthogonal to the tier (a healthy prediction can still land
in UNCERTAIN_TOP2 if the model isn't sure).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from app.ml.predictor import ClassScore

NOT_DURIAN_CLASS = "not_durian"
HEALTHY_CLASS = "Healthy"


class DiagnosisTier(str, Enum):
    CONFIDENT = "confident"
    UNCERTAIN_TOP2 = "uncertain_top2"
    NOT_CONFIDENT = "not_confident"
    NOT_DURIAN = "not_durian"


@dataclass
class Diagnosis:
    tier: DiagnosisTier
    top1: ClassScore
    top2: Optional[ClassScore]
    top3: List[ClassScore]
    is_healthy: bool


def classify(
    top3: List[ClassScore], confidence_high: float, confidence_low: float
) -> Diagnosis:
    if not top3:
        raise ValueError("top3 must contain at least one prediction")
    if not confidence_low <= confidence_high:
        raise ValueError("confidence_low must be <= confidence_high")

    top1 = top3[0]
    top2 = top3[1] if len(top3) > 1 else None
    is_healthy = top1.label == HEALTHY_CLASS

    if top1.label == NOT_DURIAN_CLASS:
        return Diagnosis(DiagnosisTier.NOT_DURIAN, top1, top2, top3, False)
    if top1.confidence < confidence_low:
        return Diagnosis(DiagnosisTier.NOT_CONFIDENT, top1, top2, top3, is_healthy)
    if top1.confidence < confidence_high:
        return Diagnosis(DiagnosisTier.UNCERTAIN_TOP2, top1, top2, top3, is_healthy)
    return Diagnosis(DiagnosisTier.CONFIDENT, top1, top2, top3, is_healthy)
