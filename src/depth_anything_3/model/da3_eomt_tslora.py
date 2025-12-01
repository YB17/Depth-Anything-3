"""Bridging utilities for DA3 + EoMT with token-selective LoRA.

This module provides lightweight helpers to assemble LoRA token masks and
attention block masks that respect the `[G, S]` token ordering described in the
integration design. The helpers are intentionally minimal so the existing DA3
and EoMT code paths remain untouched unless explicitly enabled by callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn

from depth_anything_3.model.dinov2.layers.attention import Attention

from third_party.eomt.models.eomt import EoMT
from third_party.eomt.models.scale_block import ScaleBlock


@dataclass
class SegmentationTokenLayout:
    """Describe the layout of geometry and segmentation tokens."""

    geom_tokens: int
    seg_queries: int

    @property
    def total(self) -> int:
        return self.geom_tokens + self.seg_queries


class DepthAnything3EoMTTSLoRA(nn.Module):
    """Lightweight DA3 + EoMT TS-LoRA bridge.

    The class bundles helpers for token-selective LoRA masks, block attention
    masks, and a segmentation head that mirrors the original EoMT design
    (class head, mask head, multi-scale upscaling). It intentionally avoids
    changing DA3 defaults and is only exercised when explicitly invoked by
    segmentation-aware code paths.
    """

    def __init__(
        self,
        num_seg_queries: int,
        embed_dim: int,
        num_prefix_tokens: int,
        patch_size: Tuple[int, int],
        num_upscale_blocks: Optional[int] = None,
        hidden_mult: int = 1,
    ):
        super().__init__()
        self.num_seg_queries = num_seg_queries
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens

        self.seg_query_embed = nn.Embedding(num_seg_queries, embed_dim)
        self.class_head = nn.Linear(embed_dim, hidden_mult * embed_dim)
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        max_patch = max(patch_size[0], patch_size[1])
        if num_upscale_blocks is None:
            num_upscale_blocks = max(1, int(torch.tensor(max_patch).log2().item()) - 2)
        self.upscale = nn.Sequential(*[ScaleBlock(embed_dim) for _ in range(num_upscale_blocks)])

    def build_token_mask(self, layout: SegmentationTokenLayout, batch: int, device) -> Tensor:
        """Create a token mask that activates LoRA only for segmentation queries."""

        mask = torch.zeros(batch, layout.total, device=device)
        mask[:, layout.geom_tokens :] = 1.0
        return mask

    def build_block_attention_mask(self, layout: SegmentationTokenLayout, batch: int, device) -> Tensor:
        """Construct the block attention mask enforcing the `[G, S]` visibility rule."""

        attn_mask = torch.zeros(batch, layout.total, layout.total, device=device, dtype=torch.bool)
        attn_mask[:, : layout.geom_tokens, layout.geom_tokens :] = True
        return attn_mask

    def forward(
        self,
        tokens: Tensor,
        layout: SegmentationTokenLayout,
        patch_grid: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """Run the segmentation heads on `[G, S]` tokens.

        Args:
            tokens: Full token sequence shaped ``(B, N_total, C)`` where the last
                ``layout.seg_queries`` entries correspond to segmentation
                queries. Geometry tokens come first.
            layout: Token layout descriptor.
            patch_grid: ``(H, W)`` patch grid used to reshape patch tokens back
                to feature maps.

        Returns:
            mask_logits: ``(B, K, H_mask, W_mask)`` mask logits.
            class_logits: ``(B, K, hidden_mult*embed_dim)`` class logits.
        """

        seg_queries = tokens[:, layout.geom_tokens :, :]
        class_logits = self.class_head(seg_queries)

        patch_tokens = tokens[:, self.num_seg_queries + self.num_prefix_tokens :, :]
        patch_map = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], -1, *patch_grid)
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(seg_queries), self.upscale(patch_map)
        )

        return mask_logits, class_logits


class DA3EoMTTSLoRANetwork(nn.Module):
    """Minimal end-to-end scaffold wiring DA3 encoders to EoMT heads.

    This wrapper is intentionally lightweight: it accepts a DA3 encoder callable
    (``encoder_fn``) that returns a tuple of ``(tokens, patch_grid, layout)``
    when invoked on normalized images. The returned tokens must already follow
    the ``[G, S]`` ordering so the segmentation head can slice out query tokens
    and apply the EoMT-style heads defined above. All attention masks and
    token-selective LoRA masks should therefore be injected by the encoder via
    the provided layout.

    The wrapper mirrors ``third_party.eomt.models.eomt.EoMT`` so it can be used
    inside the existing Lightning training stack and leverage the original
    annealing behaviour. Depth/ray/pose heads can be composed inside
    ``encoder_fn``; this class simply forwards segmentation logits expected by
    EoMT losses.
    """

    def __init__(
        self,
        encoder_fn: Callable[
            [Tensor], Tuple[Tensor, Tuple[int, int], SegmentationTokenLayout, dict]
        ],
        embed_dim: int,
        num_classes: int,
        num_seg_queries: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        tslora_cfg: Optional[dict] | None = None,
    ):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.tslora_cfg = tslora_cfg or {}
        self.seg_head = DepthAnything3EoMTTSLoRA(
            num_seg_queries=num_seg_queries,
            embed_dim=embed_dim,
            num_prefix_tokens=0,
            patch_size=(1, 1),
        )
        self.class_predictor = nn.Linear(embed_dim, num_classes + 1)
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.last_aux: dict = {}

    def _run_encoder(self, images: Tensor, **kwargs):
        tokens, patch_grid, layout, aux = self.encoder_fn(images, **kwargs)
        aux = aux or {}
        mask_logits, class_feats = self.seg_head(tokens, layout, patch_grid)
        class_logits = self.class_predictor(class_feats)
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "aux": aux,
        }

    def forward(self, images: Tensor, **kwargs):
        outputs = self._run_encoder(images, **kwargs)
        self.last_aux = outputs["aux"]
        return [outputs["mask_logits"]], [outputs["class_logits"]]

    def forward_with_segmentation(self, images: Tensor, **kwargs) -> dict:
        """Forward pass that exposes segmentation and auxiliary geometry outputs."""

        outputs = self._run_encoder(images, **kwargs)
        full = {
            "mask_logits": outputs["mask_logits"],
            "class_logits": outputs["class_logits"],
        }
        full.update(outputs.get("aux", {}))
        return full

    def inference_with_segmentation(self, images: Tensor, **kwargs) -> dict:
        """Alias used by the API to surface segmentation alongside depth/rays."""

        return self.forward_with_segmentation(images, **kwargs)


class SimpleDA3Encoder(nn.Module):
    """Lightweight placeholder encoder to unblock config-driven training.

    The encoder projects images to a coarse patch grid, appends learned
    segmentation queries, and emits a ``SegmentationTokenLayout`` so downstream
    heads can build masks without relying on the full DA3 backbone. This keeps
    the training/CLI flow executable until a full DA3-backed encoder is wired
    in.
    """

    def __init__(
        self,
        embed_dim: int,
        num_seg_queries: int,
        patch_stride: int = 16,
        tslora_cfg: Optional[dict] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_seg_queries = num_seg_queries
        self.patch_stride = patch_stride
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_stride, stride=patch_stride)
        self.seg_queries = nn.Embedding(num_seg_queries, embed_dim)
        # Minimal self-attention layer to exercise block masks and token-selective LoRA.
        tslora_cfg = tslora_cfg or {
            "enable": True,
            "apply_to": ["q", "k", "v", "ffn_in", "ffn_out"],
            "rank": 4,
            "alpha": 1.0,
        }
        self.self_attn = Attention(
            embed_dim,
            num_heads=4,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            tslora_cfg=tslora_cfg,
        )

    def forward(
        self, images: Tensor, **kwargs
    ) -> Tuple[Tensor, Tuple[int, int], SegmentationTokenLayout, dict]:
        b, _, h, w = images.shape
        patch_tokens = self.proj(images)
        patch_grid = patch_tokens.shape[-2:]
        geom_tokens = patch_tokens.flatten(2).transpose(1, 2)  # (B, G, C)
        seg = self.seg_queries.weight.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([geom_tokens, seg], dim=1)
        layout = SegmentationTokenLayout(
            geom_tokens=geom_tokens.shape[1], seg_queries=self.num_seg_queries
        )

        # Build token-selective LoRA mask: only segmentation queries receive LoRA updates.
        token_mask = torch.zeros(b, layout.total, device=images.device)
        token_mask[:, layout.geom_tokens :] = 1.0

        # Block attention mask enforcing G->S masking.
        attn_mask = torch.zeros(b, layout.total, layout.total, device=images.device, dtype=torch.bool)
        attn_mask[:, : layout.geom_tokens, layout.geom_tokens :] = True

        tokens = self.self_attn(tokens, attn_mask=attn_mask, token_mask=token_mask)
        aux = {"patch_grid": patch_grid, "token_mask": token_mask, "attn_mask": attn_mask}
        return tokens, patch_grid, layout, aux


def build_da3_eomt_network_from_config(cfg: dict, num_classes: Optional[int] = None) -> EoMT:
    """Factory to assemble a DA3-backed EoMT network from YAML-style config.

    The builder accepts an optional ``encoder_fn`` entry (callable or import
    path). When absent, a lightweight placeholder encoder is used so training
    can proceed with the provided config templates.
    """

    embed_dim = cfg.get("embed_dim", 768)
    num_seg_queries = cfg.get("num_seg_queries", 128)
    num_blocks = cfg.get("num_blocks", 4)
    masked_attn = cfg.get("masked_attn_enabled", True)

    tslora_cfg = {
        "enable": cfg.get("enable_tslora", cfg.get("tslora_enable", False)),
        "rank": cfg.get("tslora_rank", cfg.get("lora_rank", 4)),
        "alpha": cfg.get("tslora_alpha", cfg.get("lora_alpha", 1.0)),
        "apply_to": cfg.get(
            "tslora_apply_to", cfg.get("lora_apply_to", ["q", "k", "v", "ffn_in", "ffn_out"])
        ),
    }

    encoder_fn = cfg.get("encoder_fn")
    if isinstance(encoder_fn, str):
        module_name, fn_name = encoder_fn.rsplit(":", 1)
        encoder_fn = getattr(__import__(module_name, fromlist=[fn_name]), fn_name)
    if encoder_fn is None:
        encoder = SimpleDA3Encoder(
            embed_dim=embed_dim,
            num_seg_queries=num_seg_queries,
            tslora_cfg=tslora_cfg,
        )
        encoder_fn = encoder

    if num_classes is None:
        num_classes = cfg.get("num_classes", 0)

    scaffold = DA3EoMTTSLoRANetwork(
        encoder_fn=encoder_fn,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_seg_queries=num_seg_queries,
        num_blocks=num_blocks,
        masked_attn_enabled=masked_attn,
        tslora_cfg=tslora_cfg,
    )

    model = EoMT(
        encoder=scaffold,  # EoMT expects ``encoder.backbone``; scaffold keeps minimal surface.
        num_classes=num_classes,
        num_q=num_seg_queries,
        num_blocks=num_blocks,
        masked_attn_enabled=masked_attn,
    )
    model.tslora_cfg = tslora_cfg
    return model
