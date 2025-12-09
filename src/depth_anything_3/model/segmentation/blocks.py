from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# from depth_anything_3.model.dinov2.vision_transformer import (
#     DropPath,
#     LayerScale,
#     drop_add_residual_stochastic_depth,
# )
from depth_anything_3.model.dinov2.layers.drop_path import DropPath
from depth_anything_3.model.dinov2.layers.layer_scale import LayerScale
from depth_anything_3.model.dinov2.layers.block import drop_add_residual_stochastic_depth

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
        drop: float = 0.0,
        ln_eps: float = 1e-6,
        init_values: float | None = None,
        drop_path: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ls1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=drop)
        self.ls2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        geom_tokens: Tensor,
        bottleneck_tokens: Tensor,
        attn_mask: Optional[Tensor] = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tensor:
        _ = patch_grid or self.patch_grid

        def attn_residual_func(
            b_tokens: Tensor, geom_tokens: Tensor, attn_mask: Optional[Tensor] = None
        ) -> Tensor:
            attn_out, _ = self.mha(
                query=self.norm1(b_tokens),
                key=geom_tokens,
                value=geom_tokens,
                attn_mask=attn_mask,
                need_weights=False,
            )
            return self.ls1(attn_out)

        def ffn_residual_func(b_tokens: Tensor) -> Tensor:
            return self.ls2(self.ffn(self.norm2(b_tokens)))

        if self.training and self.sample_drop_ratio > 0.1:
            bottleneck_tokens = drop_add_residual_stochastic_depth(
                bottleneck_tokens,
                residual_func=lambda x_subset: attn_residual_func(
                    x_subset, geom_tokens, attn_mask=attn_mask
                ),
                sample_drop_ratio=self.sample_drop_ratio,
            )
            bottleneck_tokens = drop_add_residual_stochastic_depth(
                bottleneck_tokens,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            bottleneck_tokens = bottleneck_tokens + self.drop_path1(
                attn_residual_func(bottleneck_tokens, geom_tokens, attn_mask=attn_mask)
            )
            bottleneck_tokens = bottleneck_tokens + self.drop_path2(
                ffn_residual_func(bottleneck_tokens)
            )
        else:
            bottleneck_tokens = bottleneck_tokens + attn_residual_func(
                bottleneck_tokens, geom_tokens, attn_mask=attn_mask
            )
            bottleneck_tokens = bottleneck_tokens + ffn_residual_func(bottleneck_tokens)
        return bottleneck_tokens


class GSegFromBBlock(nn.Module):
    """Update segmentation geometry from bottleneck tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        drop: float = 0.0,
        ln_eps: float = 1e-6,
        init_values: float | None = None,
        drop_path: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ls1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=drop)
        self.ls2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        seg_tokens: Tensor,
        bottleneck_tokens: Tensor,
        attn_mask: Optional[Tensor] = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tensor:
        _ = patch_grid or self.patch_grid

        def attn_residual_func(
            g_tokens: Tensor, b_tokens: Tensor, attn_mask: Optional[Tensor] = None
        ) -> Tensor:
            attn_out, _ = self.mha(
                query=self.norm1(g_tokens),
                key=b_tokens,
                value=b_tokens,
                attn_mask=attn_mask,
                need_weights=False,
            )
            return self.ls1(attn_out)

        def ffn_residual_func(g_tokens: Tensor) -> Tensor:
            return self.ls2(self.ffn(self.norm2(g_tokens)))

        if self.training and self.sample_drop_ratio > 0.1:
            seg_tokens = drop_add_residual_stochastic_depth(
                seg_tokens,
                residual_func=lambda x_subset: attn_residual_func(
                    x_subset, bottleneck_tokens, attn_mask=attn_mask
                ),
                sample_drop_ratio=self.sample_drop_ratio,
            )
            seg_tokens = drop_add_residual_stochastic_depth(
                seg_tokens,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            seg_tokens = seg_tokens + self.drop_path1(
                attn_residual_func(seg_tokens, bottleneck_tokens, attn_mask=attn_mask)
            )
            seg_tokens = seg_tokens + self.drop_path2(ffn_residual_func(seg_tokens))
        else:
            seg_tokens = seg_tokens + attn_residual_func(
                seg_tokens, bottleneck_tokens, attn_mask=attn_mask
            )
            seg_tokens = seg_tokens + ffn_residual_func(seg_tokens)
        return seg_tokens


class SemanticBlock(nn.Module):
    """Decoder-style block where queries attend to segmentation geometry."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        drop: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
        ln_eps: float = 1e-6,
        init_values: float | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ls1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.ffn = _FFN(embed_dim, mlp_ratio=mlp_ratio, dropout=drop)
        self.ls2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.patch_grid = patch_grid
        self.num_heads = num_heads

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        queries: Tensor,
        seg_tokens: Tensor,
        attn_mask: Optional[Tensor] = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tensor:
        _ = patch_grid or self.patch_grid

        def attn_residual_func(
            q_tokens: Tensor, seg_tokens: Tensor, attn_mask: Optional[Tensor] = None
        ) -> Tensor:
            attn_out, _ = self.mha(
                query=self.norm1(q_tokens),
                key=seg_tokens,
                value=seg_tokens,
                attn_mask=attn_mask,
                need_weights=False,
            )
            return self.ls1(attn_out)

        def ffn_residual_func(q_tokens: Tensor) -> Tensor:
            return self.ls2(self.ffn(self.norm2(q_tokens)))

        if self.training and self.sample_drop_ratio > 0.1:
            queries = drop_add_residual_stochastic_depth(
                queries,
                residual_func=lambda x_subset: attn_residual_func(
                    x_subset, seg_tokens, attn_mask=attn_mask
                ),
                sample_drop_ratio=self.sample_drop_ratio,
            )
            queries = drop_add_residual_stochastic_depth(
                queries,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            queries = queries + self.drop_path1(
                attn_residual_func(queries, seg_tokens, attn_mask=attn_mask)
            )
            queries = queries + self.drop_path2(ffn_residual_func(queries))
        else:
            queries = queries + attn_residual_func(queries, seg_tokens, attn_mask=attn_mask)
            queries = queries + ffn_residual_func(queries)
        return queries


class SegmentationLayer(nn.Module):
    """Bundle bottleneck, geometry update, and semantic query update."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        drop: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
        ln_eps: float = 1e-6,
        init_values: float | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.b_block = BottleneckBlock(
            embed_dim,
            num_heads,
            mlp_ratio,
            attn_dropout,
            drop,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
            patch_grid=patch_grid,
        )
        self.g_block = GSegFromBBlock(
            embed_dim,
            num_heads,
            mlp_ratio,
            attn_dropout,
            drop,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
            patch_grid=patch_grid,
        )
        self.s_block = SemanticBlock(
            embed_dim,
            num_heads,
            mlp_ratio,
            attn_dropout,
            drop,
            patch_grid=patch_grid,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
        )

    def forward(
        self,
        geom_tokens: Tensor,
        bottleneck_tokens: Tensor,
        seg_tokens: Tensor,
        queries: Tensor,
        attn_mask: Optional[Tensor] = None,
        patch_grid: Tuple[int, int] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bottleneck_tokens = self.b_block(geom_tokens, bottleneck_tokens)
        seg_tokens = self.g_block(seg_tokens, bottleneck_tokens)
        queries = self.s_block(
            queries,
            seg_tokens,
            attn_mask=attn_mask,
            patch_grid=patch_grid,
        )
        return bottleneck_tokens, seg_tokens, queries


class SegAdapterLayer(nn.Module):
    """A shared lightweight adapter stacking B, G_seg, and S updates."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        drop: float = 0.0,
        ln_eps: float = 1e-6,
        init_values: float | None = None,
        drop_path: float = 0.0,
        patch_grid: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.b_block = BottleneckBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            drop=drop,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
            patch_grid=patch_grid,
        )
        self.g_block = GSegFromBBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            drop=drop,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
            patch_grid=patch_grid,
        )
        self.s_block = SemanticBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            drop=drop,
            patch_grid=patch_grid,
            ln_eps=ln_eps,
            init_values=init_values,
            drop_path=drop_path,
        )

    def forward(
        self,
        geom_tokens_seg: Tensor,
        bottleneck_tokens: Tensor,
        query_tokens: Tensor,
        attn_masks: dict[str, Optional[Tensor]] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        attn_masks = attn_masks or {}
        m_b = attn_masks.get("b", None)
        m_g = attn_masks.get("g", None)
        m_s = attn_masks.get("s", None)

        bottleneck_tokens = self.b_block(
            geom_tokens=geom_tokens_seg, bottleneck_tokens=bottleneck_tokens, attn_mask=m_b
        )
        geom_tokens_seg = self.g_block(
            geom_tokens=geom_tokens_seg, bottleneck_tokens=bottleneck_tokens, attn_mask=m_g
        )
        query_tokens = self.s_block(queries=query_tokens, seg_tokens=geom_tokens_seg, attn_mask=m_s)
        return geom_tokens_seg, bottleneck_tokens, query_tokens
