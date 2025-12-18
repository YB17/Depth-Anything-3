import logging
import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn

from depth_anything_3.cfg import load_config, create_object
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.model.lora.token_selective_lora import TokenSelectiveLoRALinear


def _default_da3_cfg() -> str:
    return str(MODEL_REGISTRY.get("da3-base"))


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear with merge/unmerge support."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, init_scale: float = 0.01):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r) if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=init_scale)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.merged = False

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        base_out = self.base(x)
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return base_out
        lora_out = self.lora_dropout(x) @ self.lora_A.t()
        lora_out = lora_out @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def merge(self):
        if self.merged or self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return
        self.base.weight.data += (self.lora_B @ self.lora_A).data * self.scaling
        self.merged = True

    def unmerge(self):
        if not self.merged or self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return
        self.base.weight.data -= (self.lora_B @ self.lora_A).data * self.scaling
        self.merged = False


def _parse_layer_range(layers: str, depth: int) -> List[int]:
    if layers == "all":
        return list(range(depth))
    if layers.startswith("last"):
        try:
            n = int(layers.replace("last", ""))
        except Exception:
            n = depth
        return list(range(max(depth - n, 0), depth))
    if layers.startswith("range:"):
        try:
            start, end = layers.split("range:", 1)[1].split("-")
            s, e = int(start), int(end)
            s = max(s, 0)
            e = min(e, depth - 1)
            return list(range(s, e + 1))
        except Exception:
            return list(range(depth))
    return list(range(depth))


def inject_lora_into_da3(vit: DinoVisionTransformer, lora_cfg: dict) -> Dict[str, int]:
    """Inject LoRA modules into a DA3 ViT backbone."""

    if not lora_cfg.get("enabled", False):
        return {"layers": 0, "modules": 0, "trainable_lora": 0}

    depth = len(vit.blocks)
    target_layers = _parse_layer_range(lora_cfg.get("layers", "last6"), depth)
    target = lora_cfg.get("target", "qv_ffn")
    use_qkv = "qkv" in target or "qv" in target
    use_ffn = "ffn" in target
    r = int(lora_cfg.get("r", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.05))
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    train_bias = lora_cfg.get("train_bias", "none")

    modules = 0
    if "qv" in target and not hasattr(vit.blocks[0].attn, "q_proj"):
        logging.info("[LoRA] fused qkv detected; falling back to qkv injection for attention")
    for idx in target_layers:
        blk = vit.blocks[idx]
        if use_qkv:
            qkv = blk.attn.qkv
            if isinstance(qkv, TokenSelectiveLoRALinear):
                base_linear = qkv.base
            else:
                base_linear = qkv
            lora_layer = LoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout, init_scale=init_scale)
            blk.attn.qkv = lora_layer
            modules += 1
        if use_ffn:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout, init_scale=init_scale)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout, init_scale=init_scale)
            modules += 2

    trainable = sum(p.numel() for n, p in vit.named_parameters() if "lora" in n)
    if train_bias == "all":
        for name, param in vit.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif train_bias == "lora_only":
        for name, param in vit.named_parameters():
            if "bias" in name and "lora" in name:
                param.requires_grad = True

    logging.info(
        f"[LoRA] Injected into blocks {target_layers} "
        f"(target={target}, r={r}, alpha={alpha}, dropout={dropout}); "
        f"modules={modules}"
    )
    return {"layers": len(target_layers), "modules": modules, "trainable_lora": trainable}


class _PatchEmbedShim(nn.Module):
    def __init__(self, vit: DinoVisionTransformer):
        super().__init__()
        self.vit = vit
        self.patch_size = vit.patch_embed.patch_size
        self.grid_size = vit.patch_embed.patches_resolution

    def forward(self, x: Tensor) -> Tensor:
        self._last_hw = x.shape[-2:]
        return self.vit.patch_embed(x)


class _BackboneShim(nn.Module):
    """Expose a DA3 ViT backbone with an EoMT-compatible surface."""

    def __init__(self, vit: DinoVisionTransformer):
        super().__init__()
        self.vit = vit
        self.patch_embed = _PatchEmbedShim(vit)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.pos_embed = vit.pos_embed
        self.num_register_tokens = getattr(vit, "num_register_tokens", 0)
        self.num_prefix_tokens = 1 + self.num_register_tokens

    def _pos_embed(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos = self.vit.interpolate_pos_encoding(x, self.patch_embed._last_hw[0], self.patch_embed._last_hw[1])
        x = x + pos
        if self.vit.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.vit.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x


class DA3BackboneAdapter(nn.Module):
    """Adapter exposing a DA3 backbone for EoMT while retaining depth outputs."""

    def __init__(
        self,
        da3_config_path: str = "",
        da3_ckpt_path: str = "",
        lora: Optional[dict] = None,
        freeze_depth_head: bool = False,
    ):
        super().__init__()
        self.da3_config_path = da3_config_path or _default_da3_cfg()
        self.da3_ckpt_path = da3_ckpt_path
        cfg = load_config(self.da3_config_path)
        self.da3: DepthAnything3Net = create_object(cfg)
        if da3_ckpt_path:
            self._load_ckpt(da3_ckpt_path)

        self.backbone = _BackboneShim(self.da3.backbone.pretrained)
        self.pixel_mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1), requires_grad=False)
        self.pixel_std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1), requires_grad=False)

        self.lora_report: Dict[str, int] = {"layers": 0, "modules": 0, "trainable_lora": 0}
        if lora is None:
            lora = {}
        if lora.get("enabled", False):
            self.lora_report = inject_lora_into_da3(self.da3.backbone.pretrained, lora)

        self._block_param_to_idx: Dict[int, int] = {}
        for idx, blk in enumerate(self.backbone.blocks):
            for p in blk.parameters():
                self._block_param_to_idx[id(p)] = idx

        self._depth_head_params = {id(p) for p in self.da3.head.parameters()}

        if freeze_depth_head:
            for p in self.da3.head.parameters():
                p.requires_grad = False

    def _load_ckpt(self, path: str):
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        incompatible = self.da3.load_state_dict(state, strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        if missing:
            logging.info(f"[DA3BackboneAdapter] Missing keys while loading ckpt: {missing[:10]}")
        if unexpected:
            logging.info(f"[DA3BackboneAdapter] Unexpected keys while loading ckpt: {unexpected[:10]}")

    def forward(self, x: Tensor, **kwargs):
        raise RuntimeError("DA3BackboneAdapter is not a standalone forward; use within EoMT.")

    def forward_depth(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x[:, None, ...]
        output = self.da3(x)
        depth = output.get("depth", None) if isinstance(output, dict) else getattr(output, "depth", None)
        if depth is None:
            raise RuntimeError("DA3 model did not return depth.")
        if depth.dim() == 5:
            depth = depth.view(-1, *depth.shape[2:])
        return depth

    def get_vit_blocks(self) -> List[nn.Module]:
        return list(self.backbone.blocks)

    def get_encoder_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.backbone.named_parameters():
            yield f"backbone.{name}", param

    def get_depth_head_params(self) -> Iterator[nn.Parameter]:
        return self.da3.head.parameters()

    def block_index_for_param(self, param: nn.Parameter) -> Optional[int]:
        return self._block_param_to_idx.get(id(param))

    def is_depth_head_param(self, param: nn.Parameter) -> bool:
        return id(param) in self._depth_head_params

    def build_teacher_copy(self, ckpt_path: str) -> "DA3BackboneAdapter":
        return DA3BackboneAdapter(
            da3_config_path=self.da3_config_path,
            da3_ckpt_path=ckpt_path,
            lora={"enabled": False},
            freeze_depth_head=True,
        )
