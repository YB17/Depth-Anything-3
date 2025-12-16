from __future__ import annotations

import glob
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

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
    possible_prefixes: tuple[str, ...] = ("model.backbone.pretrained.", "model.", "module.", "backbone.", "net.", ""),
) -> None:
    """
    Load DA3 pretrained weights from a safetensors file into the given backbone module.

    Args:
        backbone: The target backbone (e.g., DinoVisionTransformer instance).
        ckpt_path: Path to the `.safetensors` checkpoint file.
        strict: Whether to enforce strict loading via ``load_state_dict``.
        possible_prefixes: Collection of prefixes to try stripping from checkpoint keys.
    """
    log.info("------ load_da3_pretrained_backbone ------")
    if not ckpt_path:
        log.info("No DA3 pretrained path provided; skipping backbone initialization.")
        return
    if not os.path.isfile(ckpt_path):
        log.info("DA3 pretrained checkpoint not found at %s; skipping.", ckpt_path)
        return

    log.info("Loading DA3 pretrained backbone from %s", ckpt_path)
    state_dict = load_safetensors(ckpt_path)

    log.info("Checkpoint contains %d keys. First 10 keys:", len(state_dict))
    for i, key in enumerate(list(state_dict.keys())[:10]):
        log.info("  Checkpoint key %d: %s", i, key)

    backbone_sd = backbone.state_dict()
    
    # 🔧 添加backbone结构信息
    log.info("Backbone state_dict contains %d keys. First 10 keys:", len(backbone_sd))
    for i, key in enumerate(list(backbone_sd.keys())[:10]):
        log.info("  Backbone key %d: %s", i, key)
    
    best_match: Mapping[str, Any] | None = None
    best_hit = -1
    best_prefix = ""

    for prefix in possible_prefixes:
        candidate = _strip_prefix_if_present(state_dict, prefix)
        hits = sum(key in backbone_sd for key in candidate.keys())
        if hits > best_hit:
            best_hit = hits
            best_match = candidate
            best_prefix = prefix

    if best_match is None or best_hit == 0:
        log.warning("No overlapping keys between DA3 checkpoint and backbone; skipping load.")
        return

    log.info("Best matching prefix: '%s' with %d hits", best_prefix, best_hit)
    
    filtered = {k: v for k, v in best_match.items() if k in backbone_sd}
    missing = [k for k in backbone_sd.keys() if k not in filtered]
    unexpected = [k for k in best_match.keys() if k not in backbone_sd]

    log.info(
        "DA3 pretrained loading: %d matched keys, %d missing in checkpoint, %d unexpected in checkpoint.",
        len(filtered),
        len(missing),
        len(unexpected),
    )
    
    # 🔧 关键验证：检查是否有checkpoint中的键没有被加载
    if unexpected:
        log.warning("Found %d unexpected keys in checkpoint (not present in backbone):", len(unexpected))
        # 只显示前20个，避免日志过长
        for i, key in enumerate(unexpected[:20]):
            log.warning("  Unexpected key %d: %s", i, key)
        if len(unexpected) > 20:
            log.warning("  ... and %d more unexpected keys", len(unexpected) - 20)
        log.warning("⚠️  WARNING: These checkpoint keys were NOT loaded! Please verify if this is expected.")
        
        # 🔧 新增：保存所有unexpected keys到文件
        unexpected_keys_file = "unexpected_keys_report.txt"
        try:
            with open(unexpected_keys_file, "w") as f:
                f.write(f"Unexpected Keys Report\n")
                f.write(f"=" * 80 + "\n")
                f.write(f"Total unexpected keys: {len(unexpected)}\n")
                f.write(f"Checkpoint: {ckpt_path}\n")
                f.write(f"=" * 80 + "\n\n")
                
                # 统计unexpected keys的前缀分布
                prefix_counts = {}
                for key in unexpected:
                    # 提取更详细的前缀（例如 model.cam_dec 而不只是 model）
                    parts = key.split('.')
                    if len(parts) >= 2:
                        prefix = '.'.join(parts[:2])  # 例如 "model.cam_dec"
                    else:
                        prefix = parts[0] if parts else key
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                
                f.write("Unexpected keys by prefix:\n")
                for prefix, count in sorted(prefix_counts.items()):
                    f.write(f"  {prefix}: {count} keys\n")
                f.write("\n" + "=" * 80 + "\n\n")
                
                f.write("All unexpected keys:\n")
                for i, key in enumerate(unexpected, 1):
                    f.write(f"{i:4d}. {key}\n")
            
            log.info("✓ All unexpected keys saved to: %s", unexpected_keys_file)
        except Exception as e:
            log.warning("Failed to save unexpected keys to file: %s", e)
    else:
        log.info("✓ All checkpoint keys were successfully matched and loaded!")

    # 🔧 显示missing keys的样本（这些应该是新增的分割分支参数）
    if missing:
        log.info("Found %d missing keys in backbone (not in checkpoint, will use random initialization):", len(missing))
        for i, key in enumerate(missing[:20]):
            log.info("  Missing key %d: %s", i, key)
        if len(missing) > 20:
            log.info("  ... and %d more missing keys", len(missing) - 20)
        
        # 🔧 新增：保存所有missing keys到文件
        missing_keys_file = "missing_keys_report.txt"
        try:
            with open(missing_keys_file, "w") as f:
                f.write(f"Missing Keys Report\n")
                f.write(f"=" * 80 + "\n")
                f.write(f"Total missing keys: {len(missing)}\n")
                f.write(f"Checkpoint: {ckpt_path}\n")
                f.write(f"=" * 80 + "\n\n")
                
                # 统计missing keys的前缀分布
                prefix_counts = {}
                for key in missing:
                    prefix = key.split('.')[0] if '.' in key else key
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                
                f.write("Missing keys by prefix:\n")
                for prefix, count in sorted(prefix_counts.items()):
                    f.write(f"  {prefix}: {count} keys\n")
                f.write("\n" + "=" * 80 + "\n\n")
                
                f.write("All missing keys:\n")
                for i, key in enumerate(missing, 1):
                    f.write(f"{i:4d}. {key}\n")
            
            log.info("✓ All missing keys saved to: %s", missing_keys_file)
        except Exception as e:
            log.warning("Failed to save missing keys to file: %s", e)
        
    # 🔧 计算加载覆盖率
    total_ckpt_keys = len(best_match)
    loaded_ckpt_keys = len(filtered)
    coverage = (loaded_ckpt_keys / total_ckpt_keys * 100) if total_ckpt_keys > 0 else 0
    log.info("Checkpoint coverage: %d/%d keys loaded (%.2f%%)", loaded_ckpt_keys, total_ckpt_keys, coverage)
    
    if coverage < 90:
        log.warning("⚠️  WARNING: Less than 90%% of checkpoint keys were loaded! Please verify the model structure.")
    elif coverage == 100:
        log.info("✓ Perfect! 100%% of checkpoint keys were loaded into the backbone.")

    backbone_sd.update(filtered)
    backbone.load_state_dict(backbone_sd, strict=strict)
    log.info("DA3 pretrained backbone weights loaded (strict=%s).", strict)