from __future__ import annotations

import glob
import logging
import os
from typing import Any, Mapping

from safetensors.torch import load_file as load_safetensors
from torch import nn

log = logging.getLogger(__name__)


def _strip_prefix_if_present(state_dict: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def resolve_da3_ckpt_path(path: str) -> str:
    """Resolve a DA3 checkpoint path, optionally globbing directories."""

    if not path:
        return ""

    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not candidates:
            log.warning("No .safetensors files found in directory %s", path)
            return ""
        return candidates[0]

    return path


def load_da3_pretrained_backbone(
    backbone: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    possible_prefixes: tuple[str, ...] = ("model.", "module.", "backbone.", "net.", ""),
) -> None:
    """
    Load DA3 pretrained weights from a safetensors file into the given backbone module.

    Args:
        backbone: The target backbone (e.g., DinoVisionTransformer instance).
        ckpt_path: Path to the `.safetensors` checkpoint file.
        strict: Whether to enforce strict loading via ``load_state_dict``.
        possible_prefixes: Collection of prefixes to try stripping from checkpoint keys.
    """

    if not ckpt_path:
        log.info("No DA3 pretrained path provided; skipping backbone initialization.")
        return
    if not os.path.isfile(ckpt_path):
        log.warning("DA3 pretrained checkpoint not found at %s; skipping.", ckpt_path)
        return

    log.info("Loading DA3 pretrained backbone from %s", ckpt_path)
    state_dict = load_safetensors(ckpt_path)

    backbone_sd = backbone.state_dict()
    best_match: Mapping[str, Any] | None = None
    best_hit = -1

    for prefix in possible_prefixes:
        candidate = _strip_prefix_if_present(state_dict, prefix)
        hits = sum(key in backbone_sd for key in candidate.keys())
        if hits > best_hit:
            best_hit = hits
            best_match = candidate

    if best_match is None or best_hit == 0:
        log.warning("No overlapping keys between DA3 checkpoint and backbone; skipping load.")
        return

    filtered = {k: v for k, v in best_match.items() if k in backbone_sd}
    missing = [k for k in backbone_sd.keys() if k not in filtered]
    unexpected = [k for k in best_match.keys() if k not in backbone_sd]

    log.info(
        "DA3 pretrained loading: %d matched keys, %d missing in checkpoint, %d unexpected in checkpoint.",
        len(filtered),
        len(missing),
        len(unexpected),
    )

    backbone_sd.update(filtered)
    backbone.load_state_dict(backbone_sd, strict=strict)
    log.info("DA3 pretrained backbone weights loaded (strict=%s).", strict)

