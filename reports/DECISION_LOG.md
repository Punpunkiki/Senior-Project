# DECISION LOG — Durian Disease Classification

Every methodological choice, with: **Decision → Why (tied to *this* dataset) →
Alternatives considered → Why not the alternatives.** Tags (e.g. `[DL-SPLIT]`)
are mirrored as comments in `config.yaml` and `src/*.py`.

> Status note: this repository was generated in a sandbox **without GPU,
> without the Mendeley dataset, and without network access to Mendeley/HF**.
> The reasoning below is final; numeric thresholds that depend on the data
> (e.g. realised class balance, the abstain threshold value) are produced when
> you run the pipeline and are marked **[fill by running]** in `RESULTS.md`.

---

## Phase 1 — EDA

### [DL-FORMAT] JPEG (raw) vs PNG (cropped) as primary input
- **Decision:** Train primarily on the **raw full-frame JPEG**; keep the cropped
  PNG as a *background-reliance probe*, not the main input.
- **Why:** The project's explicit goals are (a) robustness to real-field
  messiness and (b) answering *"is it learning disease features or dataset
  artifacts?"*. Raw JPEGs preserve the noisy backgrounds that match deployment
  (farmer/web photos) and make the artifact question *testable*.
- **Alternatives:** (1) cropped PNG as primary; (2) merge both formats.
- **Why not:** (1) Removing background hides the exact shortcut we want to
  measure and widens the domain gap to background-rich OOD images; (2) merging
  confounds the format effect. Instead the PNG powers a clean ablation
  (train-JPEG/test-PNG vs reverse) that *quantifies* background reliance — set
  via `config.data.image_format`.

### [DL-RESIZE] Resize size & interpolation
- **Decision:** 224×224, **bilinear**; val/test = resize shorter side to 256 then
  center-crop 224.
- **Why:** All three backbones were ImageNet-pretrained at 224, so 224 reuses
  their spatial priors with zero positional/stride mismatch; field images here
  are far larger than 224 (median printed in EDA) so we downscale — no invented
  detail. Bilinear is the timm default the pretrained weights expect.
- **Alternatives:** 384/512; bicubic; aspect-preserving pad.
- **Why not:** Higher res = 3–5× compute for marginal gain on ~5k images and a
  single GPU; bicubic can ring on thin lesion edges; padding wastes resolution
  and the backbones expect square inputs.

### [DL-NORM] Normalization stats
- **Decision:** ImageNet mean/std. Dataset-computed stats are recorded for an
  ablation only.
- **Why:** We fine-tune ImageNet-pretrained backbones; matching the pretraining
  input distribution keeps early filters in-range and stabilises transfer on a
  small dataset.
- **Alternatives:** dataset-computed mean/std; per-image standardisation.
- **Why not:** Re-centering to dataset stats shifts inputs away from what
  pretrained filters expect (hurts early epochs) for negligible benefit on
  natural RGB; per-image standardisation also discards absolute brightness,
  which is a real disease cue here.

### [DL-GROUP] Duplicate handling / grouping
- **Decision:** Perceptual hashing (pHash, Hamming ≤ `near_dup_hamming`);
  near-duplicate clusters become atomic **groups** kept within one split.
- **Why:** A single-orchard dataset has many near-identical shots of the same
  fruit/scene. If they straddle train/test the model matches the *photo*, not
  the *disease*, inflating test accuracy. Cross-class duplicate groups are a
  direct artifact/leakage signal feeding the central question.
- **Alternatives:** exact-hash dedup; embedding-based dedup; ignore duplicates.
- **Why not:** Exact hashing misses re-encodes/crops; embedding dedup needs a
  model and is heavier; ignoring them is the leakage we must prevent.

---

## Phase 2 — Splitting & leakage

### [DL-SPLIT] Split ratio & method
- **Decision:** **70 / 15 / 15** train/val/test, **group-aware + stratified**
  (`StratifiedGroupKFold` carving test then val).
- **Why:** 70% gives each ~5k-image class enough samples to fine-tune; 15% val is
  large enough for stable macro-F1 model selection; 15% test gives a trustworthy
  final estimate. Stratify to keep all 10 classes proportional in every split;
  group-aware to honour [DL-GROUP].
- **Alternatives:** plain random; stratified-only; 80/10/10; k-fold CV.
- **Why not:** Random/stratified-only ignore near-duplicate groups → leakage;
  80/10/10 shrinks val/test too much for stable per-class metrics on 10 classes;
  full k-fold CV is the rigorous ideal but 3× training cost across 3 models on a
  single GPU is impractical for a senior-project timeline (single held-out test
  is the defensible compromise; CV noted as future work).

### [DL-OOD] OOD set quarantine
- **Decision:** The web/social-media set is **eval-only** — never in training,
  validation, or model selection.
- **Why:** It is the instrument that answers "does it generalise beyond
  Mendeley?". Any leak into training/selection destroys that measurement.
- **Alternatives:** mix some OOD into training; use OOD for early stopping.
- **Why not:** Both contaminate the one unbiased generalisation probe we have;
  the ID→OOD gap would no longer mean anything.

---

## Phase 3 — Preprocessing & augmentation

> Principle: **surgical, not maximal.** Three classes (Yellow leaf, Pink
> disease, Sooty mold) are *colour-defined*, so colour-destroying augmentations
> are actively harmful.

### [DL-AUG-ON] Augmentations enabled
- **RandomResizedCrop (scale 0.7–1.0):** mimics framing/zoom variation; forces
  attention to local lesions rather than whole-frame layout.
- **Horizontal flip (p=0.5):** leaves/branches have no left/right canonical
  orientation → free, label-preserving variation.
- **Mild rotation (±15°):** handheld field photos are roughly but not exactly
  upright.
- **Mild brightness/contrast jitter (0.2 / 0.2):** directly simulates the
  uncontrolled field lighting the proposal calls out.
- **Why these:** each maps to a real source of variation in farmer/field photos
  without altering the diagnostic signal.

### [DL-AUG-OFF] Augmentations deliberately disabled (≥2, each tied to a disease)
- **Hue & saturation jitter (OFF):** would corrupt the colour that *defines*
  **Yellow leaf** (yellowing), **Pink disease** (pink mycelium), **Sooty mold**
  (black coating). Shifting hue could turn a yellow leaf green → wrong label.
- **Vertical flip / 180° rotation (OFF):** **Stem cracking gummosis** shows gum
  running *downward* (gravity) and fruit/branches are photographed upright;
  flipping vertically creates physically impossible images.
- **RandAugment / TrivialAugment (OFF by default):** their default op pools
  include posterize/solarize/colour ops that hit the same colour cues; left as a
  config toggle for an ablation only.
- **Heavy Gaussian blur (OFF):** **Thrips** stippling and early **Anthracnose**
  specks are tiny; blur erases them.
- **Mixup/CutMix (OFF by default):** can help regularise but blends labels and
  can mix a colour-defined class with another; enabled only as a Phase-11
  experiment.

---

## Phase 4 — Model selection (3 architectures)

### [DL-MODELS] The trio
- **Decision:** `convnextv2_tiny` (modern CNN) + `swin_tiny` (hierarchical
  transformer) + `efficientnet_b0` (efficient/edge CNN).
- **Why:** They span genuinely different inductive biases, so the comparison is
  scientific, not three near-clones:
  - *ConvNeXt-V2 Tiny* — large-kernel depthwise CNN, strong **local texture**:
    lesion spots, anthracnose patterning, mealybug residue; FCMAE+IN22k
    pretraining is sample-efficient.
  - *Swin-Tiny* — windowed/shifted attention reintroduces locality+hierarchy →
    **global context** (whole-leaf yellowing, disease spread) at ~28M params,
    far more data-efficient than ViT-Base.
  - *EfficientNet-B0* — MBConv+SE, compound-scaled, ~5.3M params (≈4.0M with the
    10-class head) → the realistic **"runs on a farmer's phone"** baseline;
    anchors the deployment axis.
  - Two are ~28M (fair CNN-vs-transformer head-to-head at matched capacity); one
    is tiny (edge constraint). Expected weakness: Swin may overfit most on ~5k
    images; EfficientNet-B0 may trail on the hardest confusable pairs.
- **Alternatives:** ResNet18/50, ConvNeXt-V2-Nano, DeiT3-Small, ViT-Base.
- **Why not:** ResNets are older baselines with weaker accuracy/param trade-offs;
  Nano is a fine lighter swap but Tiny gives more capacity headroom; DeiT3-Small
  is a reasonable substitute for Swin but lacks Swin's hierarchical feature maps
  (which also make Grad-CAM cleaner). ViT-Base: see below.

### [DL-VIT] Keep or replace the proposal's ViT-Base?
- **Decision:** **Replace** `vit-base-patch16-224` with `swin_tiny`.
- **Why:** ViT-Base is ~86M params and has weak spatial inductive bias, so it is
  data-hungry; on only ~5k images it overfits and needs long schedules / heavy
  aug to compete. Swin keeps the transformer story (the comparison the team
  wants) but with locality+hierarchy that fit small data and produce spatial
  maps for interpretability.
- **Alternatives:** keep ViT-Base; DeiT3-Small; ViT-Small.
- **Why not:** Keeping ViT-Base risks an unflattering, overfit result that
  reflects data scale, not architecture; if a professor specifically expects
  ViT-Base, it is a one-line `config.yaml` swap and we document the overfitting
  risk + mitigation (linear-probe-first, heavy aug).

---

## Phase 5 — Training strategy

### [DL-PRETRAIN] Pretrained vs from scratch
- **Decision:** ImageNet-pretrained weights for all three.
- **Why:** ~5k images is 1–2 orders of magnitude too few to learn good
  low/mid-level vision features from scratch; transfer learning is the only
  defensible route at this scale.
- **Alternatives:** train from scratch; self-supervised pretrain on the data.
- **Why not:** From scratch would badly underfit/overfit and waste the timeline;
  SSL on 5k images yields weak features vs ImageNet transfer.

### [DL-FINETUNE] Fine-tuning regime
- **Decision:** **Full fine-tune with layer-wise LR decay (LLRD)** + a short
  linear-probe warmup (head-only for the first few epochs).
- **Why:** Small + noisy data: a brief head-only warmup aligns the random head
  before perturbing the backbone (prevents large early gradients from wrecking
  pretrained features); LLRD then adapts deep, task-specific layers more than
  shallow generic edge/colour filters — ideal when fine textures define classes.
- **Alternatives (with their failure mode here):**
  - *Linear probing only* — underfits; frozen filters can't specialise to durian
    lesions, capping accuracy.
  - *Naive full fine-tune, single high LR from epoch 0* — catastrophic forgetting
    of pretrained features on a small set; unstable, overfits fast.
  - *LoRA / adapters* — designed for parameter efficiency on very large models;
    at ~28M params the saving is marginal, vision-CNN adapter tooling is less
    mature, and it adds defend-in-viva complexity for no clear gain.

### [DL-LOSS] Loss function
- **Decision:** **Cross-entropy + label smoothing (0.1)**.
- **Why:** Label smoothing curbs overconfidence on a small noisy set and yields
  better-calibrated probabilities — which we *need* because Phase 7 makes
  confidence-threshold decisions.
- **Alternatives:** plain CE; focal loss; class-weighted CE.
- **Why not:** Plain CE overfits/over-confident → poor calibration; focal loss
  targets class imbalance (data is ~balanced) and tends to *worsen* calibration;
  class weighting is held in reserve for Phase 11 only if EDA shows real
  imbalance.

### [DL-OPT] Optimizer + schedule + warmup
- **Decision:** **AdamW** (wd 0.05), **cosine** decay with **3-epoch linear
  warmup**, batch 32, grad-clip 1.0, mixed precision.
- **Why:** AdamW is the standard, stable optimizer for transformers (Swin) and
  modern CNNs; warmup prevents the large, destabilising early updates that
  attention models are notoriously sensitive to; cosine gives smooth annealing
  without manual step tuning; AMP ~halves memory/time on GPU.
- **Alternatives:** SGD+momentum; OneCycle; constant LR; no warmup.
- **Why not:** SGD trains Swin poorly and needs careful LR tuning; constant LR
  leaves accuracy on the table; **no warmup** frequently destabilises Swin in
  early epochs.

### [DL-STOP] Stopping criterion
- **Decision:** Early stopping on **validation macro-F1**, patience 10, plus a
  50-epoch cap.
- **Why:** Macro-F1 weights every disease equally, so we stop on balanced
  per-class performance, not majority-class accuracy; patience avoids stopping on
  a single noisy epoch.
- **Alternatives:** fixed epoch budget; monitor val loss/accuracy.
- **Why not:** Fixed budget wastes compute or stops early per model; val loss can
  improve while macro-F1 stalls (and vice versa); accuracy hides per-class
  collapse.

---

## Phase 6 — Evaluation

### [DL-METRICS] Metric choices
- **Decision:** Report **accuracy + macro-F1 + per-class precision/recall + a
  confusion matrix per model**, plus **calibration (reliability diagram + ECE)**.
- **Why:** Accuracy alone hides per-class failure — a model can ace easy classes
  and collapse on a confusable pair while accuracy looks fine. Macro-F1 enforces
  per-class fairness even though training is balanced. The confusion matrix shows
  *which* diseases get confused (cross-check EDA pairs: Sooty mold↔Mealybug,
  Fruit rot↔Gummosis, Stem blight↔Canker). Calibration matters because Phase 7
  thresholds on confidence — an overconfident model breaks any threshold.
- **Alternatives:** accuracy-only; micro-F1; AUC.
- **Why not:** Accuracy/micro-F1 are dominated by easy classes and mask minority
  failures; multiclass AUC is less interpretable for a deployment threshold story.

---

## Phase 7 — Confidence thresholding / abstain / OOD

### [DL-THRESH] Threshold method, value selection, calibrate-then-threshold
- **Decision:** **Temperature-scale on val (calibrate), then abstain below a
  softmax-confidence threshold** chosen from the **precision/coverage trade-off**
  on val (smallest τ reaching target precision); ID-vs-OOD separation available
  as an alternative selection rule.
- **Why:** Deployment images may show none of the 10 diseases (healthy fruit,
  wrong crop, web noise) — forcing a class is wrong. Temperature scaling first so
  the threshold operates on *trustworthy* probabilities. The value is *derived*
  from a plotted precision/coverage curve, never hardcoded to 0.5.
- **Alternatives:** raw-softmax threshold; fixed 0.5; energy score; Mahalanobis
  distance; Monte-Carlo dropout.
- **Why not:** Raw softmax is miscalibrated → wrong threshold; 0.5 is arbitrary;
  energy/Mahalanobis/MC-dropout are stronger OOD detectors but add complexity and
  are harder to defend in a senior project. A **calibrated softmax threshold** is
  simple, interpretable, and defensible *first*; it sacrifices some OOD-detection
  power (softmax can still be confidently wrong on far-OOD), noted as future work.

---

## Phase 8 — Interpretability

### [DL-INTERPRET] Method per architecture + "looking at the right thing"
- **Decision:** **Grad-CAM** on the last spatial stage for the CNNs
  (ConvNeXt-V2, EfficientNet) and **Grad-CAM with a reshape transform** for Swin
  (token→spatial); plus a label-free **occlusion test** as a quantitative proxy.
- **Why:** Grad-CAM is the standard, defensible attribution for CNNs; Swin's
  hierarchical maps reshape cleanly into a spatial grid so the *same* method
  applies (fair cross-model comparison). "Looking at the right thing" = the CAM
  peak sits on the **lesion/symptom**, not background/hand/ruler; the occlusion
  test quantifies it: masking the CAM-peak patch should drop the predicted-class
  probability **far more** than masking a random patch (`cam_over_random_ratio`).
- **Alternatives:** attention rollout (Swin); Score-CAM; LIME/SHAP.
- **Why not:** Attention rollout isn't class-discriminative and isn't comparable
  to CNN Grad-CAM; Score-CAM is far slower; LIME/SHAP are perturbation-heavy and
  noisy for fine textures. (Rollout is left as an optional config alternative.)

---

## Phase 10 — Final selection & ensemble

### [DL-FINAL] Final model + ensemble decision
- **Decision (rule, value [fill by running]):** Pick on **deployment-relevant**
  grounds — prioritise **OOD macro-F1** and **calibration (ECE)**, then size and
  latency; raw ID accuracy breaks ties only. Recommend an **ensemble only if** it
  yields a materially higher OOD macro-F1 that justifies running multiple models
  on an edge device.
- **Why:** The end user is a farmer with a phone; a slightly less accurate but
  robust, well-calibrated, low-latency model beats a fragile high-ID-accuracy
  one. An ensemble multiplies inference cost/latency — only worth it if OOD
  robustness clearly improves.
- **Alternatives:** pick on ID accuracy alone; always ensemble.
- **Why not:** ID-accuracy-only rewards Mendeley-memorisation (the failure mode
  under study); always-ensembling ignores the edge cost constraint.

---

## Phase 11 — Iterative improvement (planned interventions)
Each intervention targets a *specific observed error*, not blind tuning:
- Worst confused pair (expected Sooty mold↔Mealybug) → targeted aug / hard-example
  mining for those classes.
- Mild class weighting **only if** EDA confirms imbalance.
- Test-time augmentation (flip) if it improves val macro-F1 without hurting
  calibration.
Before/after macro-F1 and confusion sub-matrix reported in `RESULTS.md`.
