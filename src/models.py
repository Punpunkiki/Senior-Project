"""
models.py — factory for the 3 architectures + fine-tuning param groups +
Grad-CAM target resolution.  ([DL-MODELS] [DL-FINETUNE])

Trio (distinct inductive biases):
  - convnextv2_tiny  : modern large-kernel CNN  -> local texture (lesions)
  - swin_tiny        : windowed/hierarchical transformer -> data-efficient
                       global context (replaces the proposal's ViT-Base)
  - efficientnet_b0  : compound-scaled mobile CNN -> edge/phone deployment
"""
from __future__ import annotations

from typing import Any, Dict, List

from .utils import get_logger, model_cfg_by_name

log = get_logger("models")


def build_model(cfg: Dict[str, Any], name: str, num_classes: int,
                pretrained: bool = True):
    """Create a timm model with a fresh `num_classes` head.

    pretrained=True downloads ImageNet weights from the HF hub (needs network).
    On an offline box pass pretrained=False to validate the architecture/code
    path only (do NOT report metrics from a randomly-initialised backbone).
    """
    import timm
    mcfg = model_cfg_by_name(cfg, name)
    model = timm.create_model(mcfg["timm_id"], pretrained=pretrained,
                              num_classes=num_classes)
    log.info("Built %s (timm_id=%s, pretrained=%s)", name, mcfg["timm_id"],
             pretrained)
    return model


# --------------------------------------------------------------------------- #
# Fine-tuning strategies ([DL-FINETUNE])
# --------------------------------------------------------------------------- #
def _is_head(param_name: str) -> bool:
    """True only for the classifier head params. Must NOT match EfficientNet's
    `conv_head` (a backbone layer) — hence the precise prefix/segment checks."""
    n = param_name
    return (n.startswith("head.") or ".head." in n
            or n.startswith("classifier") or ".classifier" in n
            or n.startswith("fc.") or ".fc." in n)


def freeze_backbone(model) -> None:
    """Linear probing: only the classifier head trains."""
    for n, p in model.named_parameters():
        p.requires_grad = _is_head(n)


def unfreeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


def build_param_groups(cfg: Dict[str, Any], model, name: str) -> List[dict]:
    """Return optimizer param groups implementing the chosen fine-tune regime.

      linear_probe : head only (call freeze_backbone first).
      full         : single backbone_lr for body, lr for head.
      full_llrd    : layer-wise LR decay (deeper = larger LR) + lr for head.
      gradual      : same groups as full_llrd; unfreezing handled in train.py.
    """
    t = cfg["train"]
    strat = t["finetune"]["strategy"]
    head_lr = t["optimizer"]["lr"]
    base_lr = t["optimizer"]["backbone_lr"]
    wd = t["optimizer"]["weight_decay"]

    if strat == "linear_probe":
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        return [{"params": params, "lr": head_lr, "weight_decay": wd}]

    if strat == "full":
        head, body = [], []
        for n, p in model.named_parameters():
            (head if _is_head(n) else body).append(p)
        return [{"params": body, "lr": base_lr, "weight_decay": wd},
                {"params": head, "lr": head_lr, "weight_decay": wd}]

    # full_llrd / gradual -> layer-wise decay
    decay = t["finetune"]["layer_lr_decay"]
    depth_map = _layer_depth_map(model, name)
    max_depth = max(depth_map.values()) if depth_map else 0
    groups: Dict[int, dict] = {}
    for n, p in model.named_parameters():
        if _is_head(n):
            lr = head_lr
            d = max_depth + 1
        else:
            d = depth_map.get(_layer_key(n, name), 0)
            lr = base_lr * (decay ** (max_depth - d))
        groups.setdefault(d, {"params": [], "lr": lr, "weight_decay": wd})
        groups[d]["params"].append(p)
    log.info("%s param groups (LLRD decay=%.2f): %d groups, head_lr=%.2g, "
             "deepest_body_lr=%.2g", name, decay, len(groups), head_lr, base_lr)
    return list(groups.values())


def _layer_key(param_name: str, name: str) -> str:
    """Coarse stage/block key used to bucket params for LLRD."""
    parts = param_name.split(".")
    # convnext/swin: stages.<i>.blocks.<j>... ; efficientnet: blocks.<i>...
    if "stages" in parts:
        i = parts.index("stages")
        return ".".join(parts[i:i + 2])
    if "blocks" in parts:
        i = parts.index("blocks")
        return ".".join(parts[i:i + 2])
    if "layers" in parts:
        i = parts.index("layers")
        return ".".join(parts[i:i + 2])
    return parts[0]  # stem / patch_embed / norm etc.


def _layer_depth_map(model, name: str) -> Dict[str, int]:
    keys = []
    for n, _ in model.named_parameters():
        if _is_head(n):
            continue
        k = _layer_key(n, name)
        if k not in keys:
            keys.append(k)
    return {k: i for i, k in enumerate(keys)}  # earlier=shallow(0)..deeper=high


# --------------------------------------------------------------------------- #
# Grad-CAM target layers + reshape transform ([DL-INTERPRET])
# --------------------------------------------------------------------------- #
def gradcam_target_layers(model, name: str) -> List:
    """Return the layer(s) whose activations Grad-CAM uses (last spatial stage)."""
    if name.startswith("convnext"):
        return [model.stages[-1].blocks[-1]]
    if name.startswith("efficientnet"):
        # timm efficientnet: conv_head / final blocks carry last spatial maps
        return [model.conv_head] if hasattr(model, "conv_head") \
            else [model.blocks[-1]]
    if name.startswith("swin"):
        return [model.layers[-1].blocks[-1].norm2]
    raise ValueError(f"no gradcam target rule for '{name}'")


def swin_reshape_transform(tensor, height: int = 7, width: int = 7):
    """Map a Swin activation to (B, C, H, W) so Grad-CAM can pool it.

    timm Swin versions differ: the last block may emit tokens (B, H*W, C) or a
    spatial map (B, H, W, C). Handle both, then move channels to dim=1.
    """
    if tensor.dim() == 4:                       # (B, H, W, C)
        result = tensor
    else:                                       # (B, H*W, C)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
