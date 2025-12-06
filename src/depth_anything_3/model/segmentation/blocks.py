from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

class _FFN(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BottleneckBlock(nn.Module):
    """Cross-attend bottleneck tokens to geometry tokens then apply FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, geom_tokens: Tensor, bottleneck_tokens: Tensor) -> Tensor:
        attn_out, _ = self.mha(query=bottleneck_tokens, key=geom_tokens, value=geom_tokens, need_weights=False)
        mid = bottleneck_tokens + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm(mid))
        return mid + self.dropout(ffn_out)


class GSegFromBBlock(nn.Module):
    """Update segmentation geometry from bottleneck tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seg_tokens: Tensor, bottleneck_tokens: Tensor) -> Tensor:
        attn_out, _ = self.mha(query=seg_tokens, key=bottleneck_tokens, value=bottleneck_tokens, need_weights=False)
        mid = seg_tokens + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm(mid))
        return mid + self.dropout(ffn_out)


class SemanticBlock(nn.Module):
    """Decoder-style block where queries attend to segmentation geometry.

    Masked attention is driven by mask logits produced outside the block (from
    the previous semantic layer). The block only *consumes* ``prev_mask_logits``
    and uses a layer-level annealing probability to decide whether to apply the
    mask for the current step.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.patch_grid = patch_grid

    def _build_attn_mask(
        self,
        prev_mask_logits: Tensor,
        mask_prob: float,
        num_heads: int,
        training: bool,
    ) -> Optional[Tensor]:
        """Construct an attention mask using previous-layer mask logits.

        A single Bernoulli draw per block decides whether to enable masking for
        the current step. If masking is disabled or the model is not in training
        mode, ``None`` is returned to fall back to plain cross-attention.
        """

        if mask_prob is None or mask_prob <= 0 or not training:
            return None

        apply_mask = torch.rand((), device=prev_mask_logits.device) < mask_prob
        if not bool(apply_mask):
            return None

        allowed = prev_mask_logits.reshape(prev_mask_logits.shape[0], prev_mask_logits.shape[1], -1) > 0
        attn_mask = (~allowed).repeat_interleave(num_heads, dim=0)
        return attn_mask

    def forward(
        self,
        queries: Tensor,
        seg_tokens: Tensor,
        prev_mask_logits: Optional[Tensor] = None,
        mask_prob: float | None = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tensor:
        _ = patch_grid or self.patch_grid
        attn_mask = None
        if prev_mask_logits is not None and mask_prob is not None:
            attn_mask = self._build_attn_mask(
                prev_mask_logits=prev_mask_logits,
                mask_prob=mask_prob,
                num_heads=self.mha.num_heads,
                training=self.training,
            )

        attn_out, _ = self.mha(
            query=queries, key=seg_tokens, value=seg_tokens, attn_mask=attn_mask, need_weights=False
        )
        mid = queries + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm(mid))
        return mid + self.dropout(ffn_out)


class SegmentationLayer(nn.Module):
    """Bundle bottleneck, geometry update, and semantic query update."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.b_block = BottleneckBlock(embed_dim, num_heads, mlp_ratio, attn_dropout, dropout)
        self.g_block = GSegFromBBlock(embed_dim, num_heads, mlp_ratio, attn_dropout, dropout)
        self.s_block = SemanticBlock(
            embed_dim, num_heads, mlp_ratio, attn_dropout, dropout, patch_grid=patch_grid
        )

    def forward(
        self,
        geom_tokens: Tensor,
        bottleneck_tokens: Tensor,
        seg_tokens: Tensor,
        queries: Tensor,
        prev_mask_logits: Optional[Tensor] = None,
        mask_prob: float | None = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bottleneck_tokens = self.b_block(geom_tokens, bottleneck_tokens)
        seg_tokens = self.g_block(seg_tokens, bottleneck_tokens)
        queries = self.s_block(
            queries,
            seg_tokens,
            prev_mask_logits=prev_mask_logits,
            mask_prob=mask_prob,
            patch_grid=patch_grid,
        )
        return bottleneck_tokens, seg_tokens, queries
