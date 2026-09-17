"""Tests for the [DL-ACTIONS] remediation layer.

Each test plants a specific defect and asserts the pipeline actually *acts*
on it -- not merely that it reported a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.actions import (Action, ActionLog, apply_data_actions,
                         assess_artifact_risk, compute_class_weights,
                         report_confused_pairs)

CLASSES = ["Anthracnose", "Canker", "Healthy", "not_durian"]


def make_cfg(**overrides):
    cfg = {
        "data": {"classes": CLASSES, "num_classes": len(CLASSES)},
        "image": {"size": 224},
        "paths": {"outputs_dir": "./outputs"},
        "actions": {
            "enabled": True,
            "report_path": "./outputs/actions.json",
            "data_quality": {"drop_corrupt": True, "drop_small_images": True,
                             "min_side_px": 64},
            "duplicates": {"fix_cross_split_leakage": True,
                           "quarantine_cross_class_dupes": True,
                           "collapse_near_dupes_in_train": True},
            "imbalance": {"auto_class_weights": True, "trigger_ratio": 1.5,
                          "method": "effective_number", "beta": 0.999,
                          "max_weight": 10.0},
            "evaluation": {"report_confused_pairs": True,
                           "top_k_confused_pairs": 5,
                           "min_confusion_rate": 0.05},
            "interpret": {"min_cam_over_random_ratio": 1.5},
        },
    }
    for key, value in overrides.items():
        cfg["actions"][key] = {**cfg["actions"].get(key, {}), **value}
    return cfg


def frame(rows):
    """rows: (path, label, split, group)"""
    return pd.DataFrame(rows, columns=["path", "label", "split", "group"])


# --------------------------------------------------------------------------- #
# Class imbalance -> weights (the rule that used to be a hand-written TODO)
# --------------------------------------------------------------------------- #
def test_balanced_data_gets_no_weights():
    cfg = make_cfg()
    df = pd.DataFrame({"label": ["Anthracnose"] * 100 + ["Canker"] * 100
                                + ["Healthy"] * 100 + ["not_durian"] * 100})
    alog = ActionLog()
    assert compute_class_weights(cfg, df, alog) is None
    assert alog.actions[0].triggered is False


def test_imbalanced_data_produces_weights():
    cfg = make_cfg()
    df = pd.DataFrame({"label": ["Anthracnose"] * 400 + ["Canker"] * 40
                                + ["Healthy"] * 200 + ["not_durian"] * 100})
    alog = ActionLog()
    weights = compute_class_weights(cfg, df, alog)

    assert weights is not None
    assert len(weights) == len(CLASSES)
    # The rare class must be up-weighted relative to the common one.
    assert weights[CLASSES.index("Canker")] > weights[CLASSES.index("Anthracnose")]
    assert alog.triggered()


def test_weights_are_clipped_to_max():
    cfg = make_cfg(imbalance={"max_weight": 2.0})
    df = pd.DataFrame({"label": ["Anthracnose"] * 5000 + ["Canker"] * 2
                                + ["Healthy"] * 500 + ["not_durian"] * 500})
    weights = compute_class_weights(cfg, df, ActionLog())
    assert max(weights) <= 2.0


def test_inverse_frequency_method_also_works():
    cfg = make_cfg(imbalance={"method": "inverse_frequency"})
    df = pd.DataFrame({"label": ["Anthracnose"] * 400 + ["Canker"] * 40
                                + ["Healthy"] * 200 + ["not_durian"] * 100})
    weights = compute_class_weights(cfg, df, ActionLog())
    assert weights[CLASSES.index("Canker")] > weights[CLASSES.index("Anthracnose")]


def test_absent_class_gets_zero_weight():
    cfg = make_cfg()
    df = pd.DataFrame({"label": ["Anthracnose"] * 300 + ["Canker"] * 50
                                + ["Healthy"] * 100})  # no not_durian at all
    weights = compute_class_weights(cfg, df, ActionLog())
    assert weights[CLASSES.index("not_durian")] == 0.0


def test_disabled_rule_returns_none():
    cfg = make_cfg(imbalance={"auto_class_weights": False})
    df = pd.DataFrame({"label": ["Anthracnose"] * 400 + ["Canker"] * 10})
    assert compute_class_weights(cfg, df, ActionLog()) is None


# --------------------------------------------------------------------------- #
# Cross-split leakage -> drop the TRAIN copy, never the test copy
# --------------------------------------------------------------------------- #
def test_leakage_drops_train_copy_and_protects_test():
    cfg = make_cfg()
    df = frame([
        ("a.jpg", "Anthracnose", "train", "g1"),
        ("b.jpg", "Anthracnose", "test", "g1"),   # same scene, leaked
        ("c.jpg", "Canker", "train", "g2"),
        ("d.jpg", "Canker", "test", "g3"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)

    assert "a.jpg" not in set(cleaned["path"])   # train copy removed
    assert "b.jpg" in set(cleaned["path"])       # test copy survives
    fired = [a for a in alog.triggered() if "leakage" in a.rule]
    assert fired and fired[0].severity == "critical"


def test_no_leakage_means_no_action():
    cfg = make_cfg()
    df = frame([
        ("a.jpg", "Anthracnose", "train", "g1"),
        ("b.jpg", "Anthracnose", "test", "g2"),
    ])
    _, alog = apply_data_actions(cfg, df, probe=False)
    leak = [a for a in alog.actions if "fix_cross_split_leakage" in a.rule]
    assert leak and leak[0].triggered is False


# --------------------------------------------------------------------------- #
# Cross-class duplicates -> quarantine from train, count in eval
# --------------------------------------------------------------------------- #
def test_contradictory_labels_are_quarantined_from_train():
    cfg = make_cfg(duplicates={"fix_cross_split_leakage": False})
    df = frame([
        ("a.jpg", "Anthracnose", "train", "g1"),
        ("b.jpg", "Canker", "train", "g1"),      # same image, two labels
        ("c.jpg", "Healthy", "train", "g2"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)

    assert set(cleaned["path"]) == {"c.jpg"}
    fired = [a for a in alog.triggered() if "cross_class" in a.rule]
    assert fired and fired[0].severity == "critical"


def test_contradictions_in_eval_are_counted_not_deleted():
    cfg = make_cfg(duplicates={"fix_cross_split_leakage": False})
    df = frame([
        ("a.jpg", "Anthracnose", "test", "g1"),
        ("b.jpg", "Canker", "test", "g1"),
        ("c.jpg", "Healthy", "train", "g2"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)

    assert {"a.jpg", "b.jpg"} <= set(cleaned["path"])   # benchmark untouched
    fired = [a for a in alog.triggered() if "cross_class" in a.rule]
    assert fired[0].details["irreducible_eval_images"] == 2


# --------------------------------------------------------------------------- #
# Near-duplicate collapse inside train
# --------------------------------------------------------------------------- #
def test_train_near_dupes_collapse_to_one():
    cfg = make_cfg()
    df = frame([
        ("a1.jpg", "Anthracnose", "train", "g1"),
        ("a2.jpg", "Anthracnose", "train", "g1"),
        ("a3.jpg", "Anthracnose", "train", "g1"),
        ("b.jpg", "Canker", "train", "g2"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)

    assert len(cleaned[cleaned["group"] == "g1"]) == 1
    assert len(cleaned) == 2
    fired = [a for a in alog.triggered() if "collapse" in a.rule]
    assert fired and fired[0].before == 4 and fired[0].after == 2


def test_disabled_rules_leave_data_untouched():
    cfg = make_cfg(duplicates={"fix_cross_split_leakage": False,
                               "quarantine_cross_class_dupes": False,
                               "collapse_near_dupes_in_train": False})
    df = frame([
        ("a1.jpg", "Anthracnose", "train", "g1"),
        ("a2.jpg", "Anthracnose", "train", "g1"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)
    assert len(cleaned) == 2
    assert alog.triggered() == []


def test_actions_disabled_globally_is_a_no_op():
    cfg = make_cfg()
    cfg["actions"]["enabled"] = False
    df = frame([("a1.jpg", "Anthracnose", "train", "g1"),
                ("a2.jpg", "Anthracnose", "train", "g1")])
    cleaned, alog = apply_data_actions(cfg, df, probe=False)
    assert len(cleaned) == 2
    assert alog.actions == []


# --------------------------------------------------------------------------- #
# Data quality on real files
# --------------------------------------------------------------------------- #
def test_corrupt_and_tiny_images_are_dropped(tmp_path):
    from PIL import Image

    good = tmp_path / "good.jpg"
    Image.new("RGB", (256, 256), (10, 120, 30)).save(good)
    tiny = tmp_path / "tiny.jpg"
    Image.new("RGB", (32, 32), (10, 120, 30)).save(tiny)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")

    cfg = make_cfg()
    df = frame([
        (str(good), "Anthracnose", "train", "g1"),
        (str(tiny), "Anthracnose", "train", "g2"),
        (str(broken), "Canker", "train", "g3"),
    ])
    cleaned, alog = apply_data_actions(cfg, df, probe=True)

    assert set(cleaned["path"]) == {str(good)}
    rules_fired = {a.rule for a in alog.triggered()}
    assert "data_quality.drop_corrupt" in rules_fired
    assert "data_quality.min_side_px" in rules_fired


# --------------------------------------------------------------------------- #
# Evaluation -> named confused pairs
# --------------------------------------------------------------------------- #
def test_confused_pairs_are_named():
    cfg = make_cfg()
    # Anthracnose is misread as Canker 30% of the time.
    cm = np.array([
        [70, 30, 0, 0],
        [2, 98, 0, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 100],
    ])
    alog = ActionLog()
    pairs = report_confused_pairs(cfg, cm, CLASSES, alog)

    assert pairs[0]["true"] == "Anthracnose"
    assert pairs[0]["predicted"] == "Canker"
    assert pairs[0]["rate"] == pytest.approx(0.30)
    assert alog.triggered()


def test_clean_confusion_matrix_fires_nothing():
    cfg = make_cfg()
    cm = np.eye(4, dtype=int) * 100
    alog = ActionLog()
    assert report_confused_pairs(cfg, cm, CLASSES, alog) == []
    assert alog.triggered() == []


def test_confused_pairs_respects_top_k():
    cfg = make_cfg(evaluation={"top_k_confused_pairs": 2})
    cm = np.array([
        [50, 20, 15, 15],
        [20, 50, 15, 15],
        [15, 15, 50, 20],
        [15, 15, 20, 50],
    ])
    pairs = report_confused_pairs(cfg, cm, CLASSES, ActionLog())
    assert len(pairs) == 2


# --------------------------------------------------------------------------- #
# Interpretability -> artifact risk
# --------------------------------------------------------------------------- #
def test_strong_cam_ratio_is_low_risk():
    cfg = make_cfg()
    alog = ActionLog()
    assert assess_artifact_risk(cfg, 3.2, alog) == "low"
    assert alog.triggered() == []


def test_weak_cam_ratio_flags_artifact_risk():
    cfg = make_cfg()
    alog = ActionLog()
    assert assess_artifact_risk(cfg, 1.05, alog) == "high"
    fired = alog.triggered()[0]
    assert fired.severity == "critical"
    assert fired.details["recommended_next_run"] == "data.image_format: png"


def test_missing_ratio_is_unknown_not_a_pass():
    """No occlusion result must never be silently treated as 'model is fine'."""
    cfg = make_cfg()
    alog = ActionLog()
    assert assess_artifact_risk(cfg, None, alog) == "unknown"
    assert alog.actions[0].severity == "warn"


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def test_action_log_saves_and_renders(tmp_path):
    alog = ActionLog()
    alog.record(Action(stage="eda", finding="f", rule="r", triggered=True,
                       action="did the thing", before=10, after=8))
    path = tmp_path / "actions.json"
    alog.save(path)

    import json
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["n_actions_taken"] == 1
    assert saved["actions"][0]["before"] == 10

    md = alog.to_markdown()
    assert "did the thing" in md
    assert "10 → 8" in md


def test_every_rule_is_recorded_even_when_it_does_not_fire():
    """The audit trail must show rules that were evaluated and passed, so the
    thesis can say what was checked, not only what was changed."""
    cfg = make_cfg()
    df = frame([("a.jpg", "Anthracnose", "train", "g1"),
                ("b.jpg", "Canker", "test", "g2")])
    _, alog = apply_data_actions(cfg, df, probe=False)
    assert len(alog.actions) >= 3
    assert any(a.triggered is False for a in alog.actions)
