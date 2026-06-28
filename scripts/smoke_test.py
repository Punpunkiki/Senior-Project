"""
smoke_test.py — validate the code path WITHOUT the dataset, GPU, or network.

Builds each architecture with random weights (pretrained=False), runs a
forward+backward pass on synthetic tensors, and exercises the split / loss /
schedule / metric / threshold / Grad-CAM-target logic. This proves the pipeline
is wired correctly; it produces NO meaningful metrics (random weights).

Run:  python scripts/smoke_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, count_params          # noqa: E402

OK, FAIL = "  [ok]", "  [FAIL]"


def main() -> int:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    errors = []

    # 1) transforms on a synthetic image -------------------------------------
    print("1) transforms")
    try:
        from PIL import Image
        from src.data import build_transforms
        img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
        for train in (True, False):
            t = build_transforms(cfg, train=train)(img)
            assert tuple(t.shape) == (3, cfg["image"]["size"], cfg["image"]["size"])
        print(OK, "train/eval transforms ->", tuple(t.shape))
    except Exception as e:
        errors.append(("transforms", e)); print(FAIL, e)

    # 2) group-stratified split on a synthetic dataframe ---------------------
    print("2) leakage-safe split")
    try:
        import copy

        import pandas as pd
        from src.data import make_splits
        scfg = copy.deepcopy(cfg)
        scfg["split"]["method"] = "group_stratified"  # exercise the algorithm
        nc = cfg["data"]["num_classes"]
        n = 600
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "path": [f"img_{i}.jpg" for i in range(n)],
            "label": rng.integers(0, nc, n),
        })
        df["label_idx"] = df["label"]
        df["group"] = rng.integers(0, 200, n)          # 200 groups (some shared)
        out = make_splits(scfg, df, cfg["seed"])
        # assert no group spans two splits
        g = out.groupby("group")["split"].nunique()
        assert (g == 1).all(), "group leakage!"
        print(OK, "no group leakage; fractions:",
              out["split"].value_counts(normalize=True).round(2).to_dict())
    except Exception as e:
        errors.append(("split", e)); print(FAIL, e)

    # 3) metrics / calibration / threshold on synthetic logits ---------------
    print("3) metrics, calibration, threshold")
    try:
        from src.evaluate import (choose_threshold, compute_metrics,
                                  expected_calibration_error, fit_temperature,
                                  softmax)
        nc = cfg["data"]["num_classes"]
        y = np.random.randint(0, nc, 300)
        logits = np.random.randn(300, nc) * 2
        logits[np.arange(300), y] += 3.0               # make it better-than-chance
        probs = softmax(logits)
        m = compute_metrics(y, probs.argmax(1), cfg["data"]["classes"])
        ece = expected_calibration_error(probs, y, cfg["eval"]["calibration_bins"])
        T = fit_temperature(logits, y)
        thr = choose_threshold(probs, y, cfg)
        print(OK, f"acc={m['accuracy']:.2f} macroF1={m['macro_f1']:.2f} "
              f"ECE={ece:.3f} T={T:.2f} tau={thr['threshold']:.2f}")
    except Exception as e:
        errors.append(("metrics", e)); print(FAIL, e)

    # 4) build each model + fwd/bwd + param groups + gradcam target ----------
    print("4) models (pretrained=False): forward/backward + LLRD + gradcam target")
    try:
        import torch
        from src.models import (build_model, build_param_groups,
                                gradcam_target_layers)
        from src.train import build_loss, build_optimizer, build_scheduler
        x = torch.randn(2, 3, cfg["image"]["size"], cfg["image"]["size"])
        yb = torch.randint(0, 10, (2,))
        for m in cfg["models"]:
            name = m["name"]
            model = build_model(cfg, name, cfg["data"]["num_classes"],
                                pretrained=False)
            groups = build_param_groups(cfg, model, name)
            opt = build_optimizer(cfg, model, name)
            sch = build_scheduler(cfg, opt, cfg["train"]["epochs"])
            crit = build_loss(cfg)
            out = model(x)
            assert out.shape == (2, cfg["data"]["num_classes"])
            loss = crit(out, yb)
            loss.backward()
            opt.step(); sch.step()
            tgt = gradcam_target_layers(model, name)
            tot, _ = count_params(model)
            print(OK, f"{name:16s} out={tuple(out.shape)} params={tot/1e6:.1f}M "
                  f"groups={len(groups)} gradcam_target_ok={bool(tgt)}")
    except Exception as e:
        errors.append(("models", e)); print(FAIL, repr(e))

    print("\n" + ("ALL SMOKE TESTS PASSED" if not errors
                  else f"{len(errors)} SECTION(S) FAILED: {[e[0] for e in errors]}"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
