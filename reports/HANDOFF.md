# HANDOFF — Durian Disease Classifier

> Read this first. It is the entry point for anyone (human or agent) picking up
> this project. Reasoning lives in `DECISION_LOG.md`; numbers live in
> `RESULTS.md`. This file is the map.

## Project

12-class durian image classifier (senior project). Central research question:
**is the model learning real disease features, or dataset artifacts?**

Classes = 10 Mendeley diseases + `Healthy` + `not_durian` (explicit reject
class, so the model can decline rather than guess on non-durian input).

- Repo: `Punpunkiki/Senior-Project` (public)
- Working branch: `claude/durian-classifier-handoff-ejwvds`
  (identical predecessor: `claude/durian-disease-classifier-aypmn0`)

Models compared:

| Model | Params | Note |
|---|---|---|
| ConvNeXt-V2-Tiny | 27.9M | modern conv baseline |
| Swin-Tiny | 27.5M | replaced the proposal's ViT-Base — see `[DL-ARCH]` |
| EfficientNet-B0 | 4.0M | efficiency / deployability anchor |

## Status

- **Code is complete and verified.** The full pipeline (train → evaluate →
  interpret → compare, all 3 models, all 12 classes) ran end-to-end on a
  synthetic mirror of the real data layout. The component smoke test
  (`scripts/smoke_test.py`) also passes.
- **No real metrics yet.** The development sandbox had no GPU, no dataset, and
  no HuggingFace access. Every slot in `reports/RESULTS.md` is
  `[[fill by running]]`.
- **Never fabricate numbers.** Placeholder values must not reach the thesis.
  Every reported figure comes from a real run on the real data.
- **Reasoning is final** in `reports/DECISION_LOG.md` — every decision recorded
  as Decision → Why → Alternatives → Why-not, tagged `[DL-*]`.

## Repo layout

```
config.yaml                     single source of truth (no magic numbers in src/)
src/
  data.py                       discovery, predefined/flat split, pHash leakage audit, transforms
  models.py                     the three architectures
  train.py                      training loop, LLRD fine-tuning
  evaluate.py                   metrics, calibration, temperature scaling, abstain logic
  interpret.py                  Grad-CAM + occlusion test (the artifact question)
  compare.py                    cross-model comparison tables
  eda.py                        dataset profiling
  utils.py
scripts/
  smoke_test.py                 component-level check
  prepare_negatives.py          builds a balanced not_durian set, excludes durian
notebooks/
  02_colab_full_run.ipynb       *** the runner ***
  01_eda.ipynb                  optional
reports/
  DECISION_LOG.md               all reasoning, tagged [DL-*]
  RESULTS.md                    metrics template, slots unfilled
  HANDOFF.md                    this file
```

## Data

Already in the user's Google Drive. The folder (its name contains "durian")
holds:

- `Train/`, `Validation/`, `Test/` — 10 disease subfolders each
- flat `Healthy/`
- flat `not_durian/` (may have nested per-fruit-type subfolders — these are
  recursed)
- a stray `.DS_Store`

`split.method: predefined` honours the existing Train/Validation/Test split for
the 10 diseases. `Healthy` and `not_durian` are flat, so they are auto-split
70/15/15 and merged into the corresponding splits.

Because `Healthy` and `not_durian` come from different cameras and orchards
than the Mendeley 10, report their counts and the **domain-shift caveat**
separately. The proposal's "4,000 images / 400 per class, balanced" claim
applies only to the 10 disease classes.

## How to run (Colab — the reliable recipe)

1. `Runtime → Disconnect and delete runtime` (factory reset).
2. Open `notebooks/02_colab_full_run.ipynb` **fresh from GitHub**, on this
   branch.
3. Set the runtime to GPU.
4. Run top-to-bottom, in order.
5. The data cell auto-detects the Drive durian folder. The Verify cell must
   show **all 12 classes** with `{train, val, test}` counts before you go on.
6. The optional Phase-4 cell is a quick pass (`epochs=15`). Use the full
   `config.yaml` (`epochs=50`) for the numbers that go in the thesis.

## Config knobs

- `data.image_format`: `jpeg` (default) | `png` | `any`.
  Use `any` if `Healthy`/`not_durian` are PNG. A warning fires if any class
  resolves to 0 images.
- `split.method` — `predefined` (default) or flat.
- `train.epochs` / `train.batch_size`.
- `train.finetune.strategy` — `full_llrd` (layer-wise LR decay).
- `threshold.*` — abstain / reject thresholds.
- `augment.*` — **colour augmentation is intentionally off**, see `[DL-AUG-OFF]`.

## Gotchas (all handled — but know them)

- **Colab dependency clashes** (`sympy … 'core'`, torchvision `_HAS_OPS`): the
  install cell pins torch / torchvision / numpy / sympy. Do **not**
  `pip install torch` yourself and do **not** blanket-upgrade packages.
- **After any runtime restart, run from the top.** The clone and `%cd` must
  happen before anything touches `src/` or `config.yaml`.
- **The clone cell skips if the repo directory already exists** — factory-reset
  the runtime to be sure you are running the latest code.
- **`not_durian` must not hide a `durian` subfolder.** Build it with
  `scripts/prepare_negatives.py`, which excludes durian explicitly. A durian
  image inside the reject class quietly poisons the whole reject-class result.

## What the next person should do

1. Run the Colab notebook end-to-end on the real data with a GPU.
2. Fill every `[[ ... ]]` slot in `RESULTS.md` from the run's outputs.
3. Read the Grad-CAM and occlusion results against the central question — the
   interpretability evidence is what makes this a research project rather than
   a leaderboard entry.
4. Disclose the cross-split leakage audit result (`outputs/leakage_audit.json`)
   whatever it says.
