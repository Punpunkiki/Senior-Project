"""Unit tests for app/diagnosis.py's confidence-tier logic.

Pure Python -- no model, no torch -- since the tiering rule is a product
decision that must be testable on its own (see the product spec's
"ตรรกะผลตรวจ" section for the source of these cutoffs).
"""
from __future__ import annotations

import pytest

from app.diagnosis import DiagnosisTier, classify
from app.ml.predictor import ClassScore

HIGH = 0.75
LOW = 0.45


def _top3(label1, conf1, label2="Canker", conf2=0.10, label3="Thrips", conf3=0.05):
    return [
        ClassScore(label1, conf1),
        ClassScore(label2, conf2),
        ClassScore(label3, conf3),
    ]


def test_confident_disease():
    d = classify(_top3("Anthracnose", 0.90), HIGH, LOW)
    assert d.tier == DiagnosisTier.CONFIDENT
    assert d.top1.label == "Anthracnose"
    assert d.is_healthy is False


def test_confident_healthy():
    d = classify(_top3("Healthy", 0.95), HIGH, LOW)
    assert d.tier == DiagnosisTier.CONFIDENT
    assert d.is_healthy is True


def test_uncertain_band_shows_top2():
    d = classify(_top3("Pink_disease", 0.60, "Sooty_mold", 0.30), HIGH, LOW)
    assert d.tier == DiagnosisTier.UNCERTAIN_TOP2
    assert d.top2 is not None
    assert d.top2.label == "Sooty_mold"


@pytest.mark.parametrize("boundary", [0.45, 0.749])
def test_uncertain_band_is_inclusive_low_exclusive_high(boundary):
    d = classify(_top3("Pink_disease", boundary), HIGH, LOW)
    assert d.tier == DiagnosisTier.UNCERTAIN_TOP2


def test_confidence_high_boundary_is_confident():
    d = classify(_top3("Pink_disease", 0.75), HIGH, LOW)
    assert d.tier == DiagnosisTier.CONFIDENT


def test_below_low_is_not_confident():
    d = classify(_top3("Stem_blight", 0.20), HIGH, LOW)
    assert d.tier == DiagnosisTier.NOT_CONFIDENT


def test_not_durian_short_circuits_regardless_of_confidence():
    d = classify(_top3("not_durian", 0.99), HIGH, LOW)
    assert d.tier == DiagnosisTier.NOT_DURIAN
    assert d.is_healthy is False


def test_not_durian_even_at_low_confidence():
    d = classify(_top3("not_durian", 0.30), HIGH, LOW)
    assert d.tier == DiagnosisTier.NOT_DURIAN


def test_single_prediction_has_no_top2():
    d = classify([ClassScore("Healthy", 0.99)], HIGH, LOW)
    assert d.top2 is None


def test_empty_predictions_raise():
    with pytest.raises(ValueError):
        classify([], HIGH, LOW)


def test_invalid_threshold_order_raises():
    with pytest.raises(ValueError):
        classify(_top3("Healthy", 0.9), confidence_high=0.4, confidence_low=0.6)
