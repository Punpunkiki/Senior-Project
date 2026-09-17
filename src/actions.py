"""
actions.py — the REMEDIATION layer: turns findings into applied changes.

Motivation ([DL-ACTIONS]).  Every earlier stage of this pipeline *measured*
things and then stopped: EDA computed a class-imbalance ratio and logged it,
the pHash audit counted cross-split duplicates and told you to "disclose this
number", and config.yaml carried `class_weights: null  # set by Phase 11`
— i.e. a human was expected to notice and act.  A finding that nothing acts
on is not a finding, it is a footnote.

This module closes that loop.  Each rule is:

    finding (measured)  ->  threshold (config)  ->  action (applied)  ->  audit record

Design rules:
  * Nothing is silently mutated.  Every action writes an `Action` record with
    before/after counts to outputs/actions.json, and renders as a markdown
    table for the thesis ("we detected X, therefore we did Y").
  * All thresholds live in config.yaml -> `actions:`; no magic numbers here.
  * Evaluation data is protected.  When a near-duplicate spans train and test
    we drop the TRAIN copy, never the test copy, so the measuring stick keeps
    its length ([DL-LEAKFIX]).
  * Any rule can be switched off in config to reproduce the un-remediated run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .utils import ensure_dir, get_logger, save_json

log = get_logger("actions")


# --------------------------------------------------------------------------- #
# Audit record
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """One finding and what was done about it."""
    stage: str            # eda | train | evaluate | interpret
    finding: str          # what was measured
    rule: str             # which config rule evaluated it
    triggered: bool       # did the threshold fire?
    action: str           # what was actually done (or "none — below threshold")
    severity: str = "info"        # info | warn | critical
    details: Dict[str, Any] = field(default_factory=dict)
    before: Optional[int] = None  # rows/images before
    after: Optional[int] = None   # rows/images after


class ActionLog:
    """Collects Action records across a run and renders the audit trail."""

    def __init__(self) -> None:
        self.actions: List[Action] = []

    def record(self, action: Action) -> Action:
        self.actions.append(action)
        level = log.warning if action.severity in ("warn", "critical") else log.info
        if action.triggered:
            level("[%s] %s -> %s", action.stage, action.finding, action.action)
        else:
            log.info("[%s] %s -> no action needed", action.stage, action.finding)
        return action

    def triggered(self) -> List[Action]:
        return [a for a in self.actions if a.triggered]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rules_evaluated": len(self.actions),
            "n_actions_taken": len(self.triggered()),
            "actions": [asdict(a) for a in self.actions],
        }

    def save(self, path: str | Path) -> None:
        ensure_dir(Path(path).parent)
        save_json(self.to_dict(), path)
        log.info("Action audit trail -> %s (%d/%d rules fired)", path,
                 len(self.triggered()), len(self.actions))

    def to_markdown(self) -> str:
        """Markdown table for RESULTS.md — the 'what we did and why' record."""
        rows = ["| Stage | Finding | Fired | Action taken | Before → After |",
                "|---|---|---|---|---|"]
        for a in self.actions:
            delta = (f"{a.before} → {a.after}"
                     if a.before is not None and a.after is not None else "—")
            rows.append(f"| {a.stage} | {a.finding} | "
                        f"{'yes' if a.triggered else 'no'} | {a.action} | {delta} |")
        return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Image probing (one pass, reused by several rules)
# --------------------------------------------------------------------------- #
def probe_images(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image once; return per-row readability and dimensions.

    Doing this in a single pass matters: the corrupt-file and too-small rules
    would otherwise each re-open the whole dataset.
    """
    readable, widths, heights = [], [], []
    for path in df["path"]:
        try:
            with Image.open(path) as im:
                im.verify()          # cheap integrity check
            with Image.open(path) as im:
                w, h = im.size       # verify() leaves the file unusable
            readable.append(True); widths.append(w); heights.append(h)
        except Exception:
            readable.append(False); widths.append(0); heights.append(0)
    out = df.copy()
    out["readable"] = readable
    out["width"] = widths
    out["height"] = heights
    out["min_side"] = np.minimum(out["width"], out["height"])
    return out


# --------------------------------------------------------------------------- #
# Rule 1-2: data quality (corrupt / too small)
# --------------------------------------------------------------------------- #
def _drop_corrupt(cfg, df: pd.DataFrame, alog: ActionLog) -> pd.DataFrame:
    rules = cfg["actions"]["data_quality"]
    n_before = len(df)
    bad = df[~df["readable"]]
    if not rules["drop_corrupt"] or bad.empty:
        alog.record(Action(
            stage="eda", finding=f"unreadable/corrupt images: {len(bad)}",
            rule="data_quality.drop_corrupt", triggered=False,
            action="none — nothing to drop" if bad.empty else "disabled in config",
            before=n_before, after=n_before))
        return df
    kept = df[df["readable"]].reset_index(drop=True)
    alog.record(Action(
        stage="eda", finding=f"unreadable/corrupt images: {len(bad)}",
        rule="data_quality.drop_corrupt", triggered=True,
        action=f"dropped {len(bad)} unreadable file(s) from the manifest",
        severity="warn",
        details={"dropped_paths": bad["path"].head(20).tolist()},
        before=n_before, after=len(kept)))
    return kept


def _drop_small(cfg, df: pd.DataFrame, alog: ActionLog) -> pd.DataFrame:
    rules = cfg["actions"]["data_quality"]
    min_side = rules["min_side_px"]
    n_before = len(df)
    small = df[(df["min_side"] > 0) & (df["min_side"] < min_side)]
    if not rules["drop_small_images"] or small.empty:
        alog.record(Action(
            stage="eda",
            finding=f"images smaller than {min_side}px on the short side: {len(small)}",
            rule="data_quality.min_side_px", triggered=False,
            action="none — nothing below the floor" if small.empty
                   else "disabled in config",
            before=n_before, after=n_before))
        return df
    kept = df.drop(small.index).reset_index(drop=True)
    alog.record(Action(
        stage="eda",
        finding=f"images smaller than {min_side}px on the short side: {len(small)}",
        rule="data_quality.min_side_px", triggered=True,
        action=(f"dropped {len(small)} image(s) too small to survive the "
                f"{cfg['image']['size']}px resize without heavy upsampling"),
        severity="warn",
        details={"min_side_px": min_side,
                 "dropped_by_class": small["label"].value_counts().to_dict()},
        before=n_before, after=len(kept)))
    return kept


# --------------------------------------------------------------------------- #
# Rule 3-5: duplicate / leakage remediation
# --------------------------------------------------------------------------- #
def _quarantine_cross_class_dupes(cfg, df: pd.DataFrame,
                                  alog: ActionLog) -> pd.DataFrame:
    """Near-identical images carrying DIFFERENT labels are label noise: at most
    one of them can be right and we cannot tell which. Remove them from TRAIN
    (they teach contradictions) and count — but keep — the ones in val/test,
    where they form an irreducible error floor worth reporting."""
    rules = cfg["actions"]["duplicates"]
    n_before = len(df)
    if not rules["quarantine_cross_class_dupes"] or "group" not in df.columns:
        alog.record(Action(
            stage="eda", finding="cross-class near-duplicates",
            rule="duplicates.quarantine_cross_class_dupes", triggered=False,
            action="disabled in config or no grouping available",
            before=n_before, after=n_before))
        return df

    per_group_labels = df.groupby("group")["label"].nunique()
    conflicted = set(per_group_labels[per_group_labels > 1].index)
    if not conflicted:
        alog.record(Action(
            stage="eda", finding="cross-class near-duplicates: 0 groups",
            rule="duplicates.quarantine_cross_class_dupes", triggered=False,
            action="none — no contradictory labels found",
            before=n_before, after=n_before))
        return df

    in_conflict = df["group"].isin(conflicted)
    is_train = df["split"] == "train" if "split" in df.columns else True
    to_drop = df[in_conflict & is_train]
    eval_left = int((in_conflict & ~is_train).sum()) if "split" in df.columns else 0

    kept = df.drop(to_drop.index).reset_index(drop=True)
    alog.record(Action(
        stage="eda",
        finding=(f"cross-class near-duplicates: {len(conflicted)} group(s) "
                 f"carry more than one label"),
        rule="duplicates.quarantine_cross_class_dupes", triggered=True,
        action=(f"quarantined {len(to_drop)} contradictory training image(s); "
                f"left {eval_left} in val/test as a reported irreducible-error "
                f"floor"),
        severity="critical",
        details={"n_conflicted_groups": len(conflicted),
                 "irreducible_eval_images": eval_left,
                 "affected_classes": to_drop["label"].value_counts().to_dict()},
        before=n_before, after=len(kept)))
    return kept


def _fix_cross_split_leakage(cfg, df: pd.DataFrame,
                             alog: ActionLog) -> pd.DataFrame:
    """A near-duplicate spanning train and val/test inflates the score. Drop the
    TRAIN copy so the evaluation sets keep their size and meaning."""
    rules = cfg["actions"]["duplicates"]
    n_before = len(df)
    if (not rules["fix_cross_split_leakage"] or "group" not in df.columns
            or "split" not in df.columns):
        alog.record(Action(
            stage="eda", finding="cross-split near-duplicate leakage",
            rule="duplicates.fix_cross_split_leakage", triggered=False,
            action="disabled in config or not a predefined split",
            before=n_before, after=n_before))
        return df

    spans = df.groupby("group")["split"].nunique()
    leaky = set(spans[spans > 1].index)
    if not leaky:
        alog.record(Action(
            stage="eda", finding="cross-split near-duplicate leakage: 0 groups",
            rule="duplicates.fix_cross_split_leakage", triggered=False,
            action="none — no group spans multiple splits",
            before=n_before, after=n_before))
        return df

    to_drop = df[(df["group"].isin(leaky)) & (df["split"] == "train")]
    kept = df.drop(to_drop.index).reset_index(drop=True)
    alog.record(Action(
        stage="eda",
        finding=(f"cross-split leakage: {len(leaky)} near-duplicate group(s) "
                 f"span Train/Val/Test"),
        rule="duplicates.fix_cross_split_leakage", triggered=True,
        action=(f"dropped {len(to_drop)} TRAIN copy(ies); val/test left intact "
                f"so the benchmark keeps its size ([DL-LEAKFIX])"),
        severity="critical",
        details={"n_leaky_groups": len(leaky),
                 "dropped_by_class": to_drop["label"].value_counts().to_dict()},
        before=n_before, after=len(kept)))
    return kept


def _collapse_train_dupes(cfg, df: pd.DataFrame, alog: ActionLog) -> pd.DataFrame:
    """Keep one image per near-duplicate group inside TRAIN. Duplicates do not
    add information; they silently re-weight whatever they depict."""
    rules = cfg["actions"]["duplicates"]
    n_before = len(df)
    if not rules["collapse_near_dupes_in_train"] or "group" not in df.columns:
        alog.record(Action(
            stage="eda", finding="near-duplicates within train",
            rule="duplicates.collapse_near_dupes_in_train", triggered=False,
            action="disabled in config or no grouping available",
            before=n_before, after=n_before))
        return df

    is_train = df["split"] == "train" if "split" in df.columns else pd.Series(
        True, index=df.index)
    train_rows = df[is_train]
    deduped = train_rows.drop_duplicates(subset="group", keep="first")
    n_removed = len(train_rows) - len(deduped)
    if n_removed == 0:
        alog.record(Action(
            stage="eda", finding="near-duplicates within train: 0",
            rule="duplicates.collapse_near_dupes_in_train", triggered=False,
            action="none — every training image is already unique",
            before=n_before, after=n_before))
        return df

    kept = pd.concat([deduped, df[~is_train]]).sort_index().reset_index(drop=True)
    alog.record(Action(
        stage="eda",
        finding=f"near-duplicates within train: {n_removed} redundant image(s)",
        rule="duplicates.collapse_near_dupes_in_train", triggered=True,
        action=(f"collapsed each near-duplicate group to one representative, "
                f"removing {n_removed} training image(s)"),
        severity="warn",
        details={"removed_by_class": (
            train_rows.loc[~train_rows.index.isin(deduped.index), "label"]
            .value_counts().to_dict())},
        before=n_before, after=len(kept)))
    return kept


# --------------------------------------------------------------------------- #
# Rule 6: class imbalance -> class weights
# --------------------------------------------------------------------------- #
def compute_class_weights(cfg, df: pd.DataFrame,
                          alog: ActionLog) -> Optional[List[float]]:
    """Return per-class loss weights when imbalance exceeds the trigger ratio.

    This is the rule that used to read `class_weights: null  # set by Phase 11`
    in config.yaml — a hand-written TODO that nothing enforced.

    Methods:
      inverse_frequency : w_c = N / (K * n_c)          — simple, can over-boost
                                                         very rare classes
      effective_number  : w_c ∝ (1-beta) / (1-beta^n_c) — Cui et al. 2019,
                                                         saturates gracefully
    """
    classes = cfg["data"]["classes"]
    rules = cfg["actions"]["imbalance"]
    counts = (df["label"].value_counts().reindex(classes).fillna(0)
              .astype(int))
    present = counts[counts > 0]
    ratio = float(present.max() / max(present.min(), 1)) if len(present) else 1.0

    if not rules["auto_class_weights"] or ratio < rules["trigger_ratio"]:
        alog.record(Action(
            stage="train",
            finding=f"class imbalance ratio (max/min) = {ratio:.2f}",
            rule="imbalance.trigger_ratio", triggered=False,
            action=(f"none — below the {rules['trigger_ratio']} trigger; "
                    f"unweighted loss keeps the objective simple")
                   if rules["auto_class_weights"] else "disabled in config",
            details={"ratio": ratio, "counts": counts.to_dict()}))
        return None

    n = counts.to_numpy(dtype=np.float64)
    safe_n = np.maximum(n, 1.0)
    if rules["method"] == "effective_number":
        beta = float(rules["beta"])
        effective = (1.0 - np.power(beta, safe_n)) / (1.0 - beta)
        raw = 1.0 / effective
    else:
        raw = len(df) / (len(classes) * safe_n)

    raw = np.where(n > 0, raw, 0.0)               # absent class -> no weight
    weights = raw / raw[raw > 0].mean()           # normalise around 1.0
    weights = np.clip(weights, 0.0, float(rules["max_weight"]))

    alog.record(Action(
        stage="train",
        finding=f"class imbalance ratio (max/min) = {ratio:.2f}",
        rule="imbalance.trigger_ratio", triggered=True,
        action=(f"computed {rules['method']} class weights and injected them "
                f"into the loss (range {weights.min():.2f}–{weights.max():.2f}, "
                f"clipped at {rules['max_weight']})"),
        severity="warn",
        details={"ratio": ratio, "method": rules["method"],
                 "weights": {c: round(float(w), 4)
                             for c, w in zip(classes, weights)},
                 "counts": counts.to_dict()}))
    return [float(w) for w in weights]


# --------------------------------------------------------------------------- #
# Rule 7: evaluation -> which class pairs actually hurt
# --------------------------------------------------------------------------- #
def report_confused_pairs(cfg, confusion: np.ndarray, classes: List[str],
                          alog: ActionLog) -> List[Dict[str, Any]]:
    """Name the specific class pairs the model confuses, so the next data
    collection round is targeted instead of 'gather more images'."""
    rules = cfg["actions"]["evaluation"]
    if not rules["report_confused_pairs"]:
        alog.record(Action(
            stage="evaluate", finding="class-pair confusion",
            rule="evaluation.report_confused_pairs", triggered=False,
            action="disabled in config"))
        return []

    cm = np.asarray(confusion, dtype=np.float64)
    support = cm.sum(axis=1, keepdims=True)
    rates = np.divide(cm, np.maximum(support, 1e-9))
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and rates[i, j] >= rules["min_confusion_rate"]:
                pairs.append({"true": classes[i], "predicted": classes[j],
                              "rate": round(float(rates[i, j]), 4),
                              "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["rate"], reverse=True)
    pairs = pairs[: rules["top_k_confused_pairs"]]

    if not pairs:
        alog.record(Action(
            stage="evaluate",
            finding=(f"no class pair confused above "
                     f"{rules['min_confusion_rate']:.0%}"),
            rule="evaluation.min_confusion_rate", triggered=False,
            action="none — no systematic pairwise confusion"))
        return []

    worst = pairs[0]
    alog.record(Action(
        stage="evaluate",
        finding=(f"{len(pairs)} class pair(s) confused above "
                 f"{rules['min_confusion_rate']:.0%}; worst: "
                 f"{worst['true']}→{worst['predicted']} at {worst['rate']:.0%}"),
        rule="evaluation.min_confusion_rate", triggered=True,
        action=("listed the offending pairs in outputs/actions.json as targeted "
                "data-collection / label-review candidates"),
        severity="warn",
        details={"pairs": pairs}))
    return pairs


# --------------------------------------------------------------------------- #
# Rule 8: interpretability -> is the model reading background?
# --------------------------------------------------------------------------- #
def assess_artifact_risk(cfg, cam_over_random_ratio: Optional[float],
                         alog: ActionLog) -> str:
    """The project's central question, expressed as a rule.

    The occlusion test occludes the Grad-CAM peak patch and a random patch,
    and reports `cam_over_random_ratio` = (confidence drop from occluding the
    CAM region) / (drop from occluding a random region).

      ratio >> 1  the highlighted region genuinely carries the evidence
      ratio ~= 1  the CAM region matters no more than any random patch, i.e.
                  the prediction rests on diffuse context, not the lesion

    Below the configured minimum this fires, because "the model may be reading
    background" is exactly the finding this whole project exists to catch, and
    it has to change what happens next rather than become a paragraph.
    """
    rules = cfg["actions"]["interpret"]
    minimum = float(rules["min_cam_over_random_ratio"])

    if cam_over_random_ratio is None:
        alog.record(Action(
            stage="interpret", finding="occlusion test produced no ratio",
            rule="interpret.min_cam_over_random_ratio", triggered=False,
            action="none — occlusion test disabled or no examples collected",
            severity="warn"))
        return "unknown"

    if cam_over_random_ratio >= minimum:
        alog.record(Action(
            stage="interpret",
            finding=(f"occlusion cam/random ratio = {cam_over_random_ratio:.2f} "
                     f"(minimum {minimum})"),
            rule="interpret.min_cam_over_random_ratio", triggered=False,
            action=("none — occluding the highlighted region hurts the "
                    "prediction more than occluding background, as it should"),
            details={"cam_over_random_ratio": cam_over_random_ratio}))
        return "low"

    alog.record(Action(
        stage="interpret",
        finding=(f"occlusion cam/random ratio = {cam_over_random_ratio:.2f} "
                 f"is below the {minimum} minimum"),
        rule="interpret.min_cam_over_random_ratio", triggered=True,
        action=("flagged ARTIFACT RISK: the highlighted region is barely more "
                "load-bearing than a random patch. Re-run the cropped-PNG "
                "ablation (data.image_format: png); if accuracy holds up with "
                "backgrounds removed the model was reading pathology, if it "
                "collapses it was reading the dataset ([DL-INTERPRET])"),
        severity="critical",
        details={"cam_over_random_ratio": cam_over_random_ratio,
                 "minimum": minimum,
                 "recommended_next_run": "data.image_format: png"}))
    return "high"


# --------------------------------------------------------------------------- #
# Driver: all data-level actions in order
# --------------------------------------------------------------------------- #
def apply_data_actions(cfg, df: pd.DataFrame,
                       alog: Optional[ActionLog] = None,
                       probe: bool = True) -> Tuple[pd.DataFrame, ActionLog]:
    """Run every data-level rule and return the remediated DataFrame.

    Order matters: quality first (a corrupt file should not reach the
    duplicate hasher), then leakage, then contradictions, then redundancy.
    """
    alog = alog or ActionLog()
    if not cfg.get("actions", {}).get("enabled", False):
        log.warning("actions.enabled is false — running WITHOUT remediation")
        return df, alog

    n_start = len(df)
    work = probe_images(df) if probe else df.copy()
    if "readable" in work.columns:
        work = _drop_corrupt(cfg, work, alog)
        work = _drop_small(cfg, work, alog)

    work = _fix_cross_split_leakage(cfg, work, alog)
    work = _quarantine_cross_class_dupes(cfg, work, alog)
    work = _collapse_train_dupes(cfg, work, alog)

    drop_cols = [c for c in ("readable", "width", "height", "min_side")
                 if c in work.columns]
    work = work.drop(columns=drop_cols).reset_index(drop=True)

    log.info("Data remediation complete: %d -> %d images (%d removed)",
             n_start, len(work), n_start - len(work))
    return work, alog


def load_action_log(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def remediate_dataframe(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Entry point used by prepare_dataframe(): group, remediate, cache.

    The result is cached because probing every image and hashing it is the
    expensive part, and train.py + evaluate.py both ask for the same frame.
    Delete outputs/splits_remediated.json to force a re-run.
    """
    if not cfg.get("actions", {}).get("enabled", False):
        return df

    cache = Path(cfg["paths"]["outputs_dir"]) / "splits_remediated.json"
    if cache.exists():
        log.info("Reusing cached remediated splits: %s (delete to re-run).",
                 cache)
        return pd.read_json(cache, orient="records")

    if "group" not in df.columns:
        from .data import assign_groups
        df = assign_groups(cfg, df)

    cleaned, alog = apply_data_actions(cfg, df)
    alog.save(cfg["actions"]["report_path"])

    ensure_dir(cache.parent)
    cleaned.to_json(cache, orient="records", force_ascii=False)
    return cleaned
