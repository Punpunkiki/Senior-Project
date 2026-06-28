# Durian Disease Classification — Modelling Core (Senior Project)

A defensible, thesis-grade pipeline for a **10-class durian disease image
classifier**, built around one central research question:

> **Is the model learning real disease features, or just memorising
> Mendeley-dataset artifacts (backgrounds / capture setup)?**

Every methodological choice is justified in **`reports/DECISION_LOG.md`**
(Decision → Why → Alternatives → Why not). Final numbers go in
**`reports/RESULTS.md`**.

The 10 classes: Anthracnose, Canker, Fruit rot, Mealybug infestation, Pink
disease, Sooty mold, Stem blight, Stem cracking gummosis, Thrips, Yellow leaf.

---

## ⚠️ Generation-environment note (read first)
This repo was generated in a sandbox with **no GPU, no dataset, and no network
access to Mendeley/HuggingFace**. Therefore:
- The **code is complete and verified to run** (a CPU synthetic smoke test
  instantiates all 3 architectures and runs a forward/backward pass).
- **No accuracy/EDA numbers are fabricated.** Every result slot in `RESULTS.md`
  is marked `[[fill by running]]` and is produced when *you* run the pipeline on
  a machine with the data + a GPU (Colab/Kaggle/local).

## The 3 models (distinct inductive biases — [DL-MODELS])
| Model | timm id | Family | Role |
|---|---|---|---|
| `convnextv2_tiny` | `convnextv2_tiny.fcmae_ft_in22k_in1k` | modern CNN | local texture (lesions) |
| `swin_tiny` | `swin_tiny_patch4_window7_224.ms_in1k` | hierarchical transformer | data-efficient global context (**replaces ViT-Base**, [DL-VIT]) |
| `efficientnet_b0` | `efficientnet_b0.ra_in1k` | efficient CNN | edge / "farmer's phone" baseline |

---

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # optional
# GPU box (Colab/Kaggle usually ship torch already):
pip install -r requirements.txt
# CPU-only box:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Get the data
```bash
python -m src.download_data            # asks before downloading (~GB) from Mendeley
# If your network blocks data.mendeley.com, download the ZIP manually from
# https://data.mendeley.com/datasets/mhjwyb5p48/1 and unzip into ./data/
```
`./data/` must end up with **10 class sub-folders**. The quarantined OOD set
(web/social images you collect) goes in **`./data_ood/`** with the same 10
folders — it is used **only** for final evaluation ([DL-OOD]).

## Run the full pipeline
```bash
# Phase 1 — EDA (real counts, resolution, colour, duplicates)
python -m src.eda                      # or open notebooks/01_eda.ipynb
# Phases 2–5 — split + train all 3 models (creates outputs/splits.json once)
python -m src.train    --all
# Phases 6,7,9 — metrics, calibration, abstain threshold, OOD gap
python -m src.evaluate --all --ood
# Phase 8 — Grad-CAM + occlusion "lesion vs background" test
python -m src.interpret --all
# Phase 10 — comparison table + plots
python -m src.compare  --all
```
Train a single model: `python -m src.train --model swin_tiny`.

## No-GPU / no-data quick check (validates the code path only)
```bash
# random-init forward/backward on tiny synthetic data — NOT for metrics
python scripts/smoke_test.py
# once you have data, a tiny real run:
python -m src.train --model efficientnet_b0 --smoke
```

## Configuration
**All hyperparameters and paths live in `config.yaml`** (no magic numbers in
`src/`). Prefer editing the config over the source. Key knobs:
`data.image_format` (jpeg↔png ablation), `split.method`, `augment.*`,
`train.finetune.strategy`, `train.loss.name`, `threshold.*`.

## Reproducibility
- Global seed (python/numpy/torch/cuda) set in `config.yaml` (`seed: 42`) via
  `src/utils.set_seed`. DataLoader workers seeded too.
- Pinned deps in `requirements.txt`.
- **Residual non-determinism:** a few CUDA ops (some interpolation/pooling
  kernels) are not bit-exact even with `cudnn.deterministic`; we set
  `use_deterministic_algorithms(warn_only=True)` rather than crash. CPU runs are
  deterministic.

## Project layout
```
config.yaml                 # single source of truth for all hyperparameters
requirements.txt
notebooks/01_eda.ipynb      # Phase 1 EDA with markdown Decision Logs
src/
  utils.py                  # seeding, config, device, metrics helpers
  data.py                   # discovery, pHash grouping, leakage-safe splits, transforms
  eda.py                    # Phase-1 analysis (shared with the notebook)
  models.py                 # 3-architecture factory, LLRD groups, Grad-CAM hooks
  train.py                  # training loop (AMP, warmup+cosine, early stop)
  evaluate.py               # metrics, confusion, calibration, temp-scaling, threshold
  interpret.py              # Grad-CAM (CNN + Swin) + occlusion test
  compare.py                # cross-model comparison table + plots
  download_data.py          # Mendeley fetch helper
scripts/smoke_test.py       # offline code-path validation (no data/GPU)
reports/
  DECISION_LOG.md           # every decision, audit-ready
  RESULTS.md                # results template (fill by running) + limitations
outputs/                    # weights, figures, confusion matrices, gradcam, json
```

## Results & reasoning
- **`reports/DECISION_LOG.md`** — every decision with alternatives and trade-offs.
- **`reports/RESULTS.md`** — comparison table, final pick, interpretability
  verdict, ID→OOD gap, limitations (placeholders until you run it).
