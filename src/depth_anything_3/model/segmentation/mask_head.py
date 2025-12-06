from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class SegMaskHead(nn.Module):
    """Predict per-query mask logits for segmentation tokens.

    The head mirrors the lightweight MLP + einsum formulation used in EoMT to
    produce spatial mask logits from query embeddings and flattened patch
    tokens.
    """

    def __init__(self, embed_dim: int, patch_grid: Tuple[int, int]):
        super().__init__()
        self.patch_grid = patch_grid
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, queries: Tensor, seg_tokens: Tensor) -> Tensor:
        """Compute mask logits from queries and segmentation tokens.

        Args:
            queries: Query tokens of shape ``(B, Q, C)``.
            seg_tokens: Segmentation geometry tokens of shape ``(B, T, C)``.

        Returns:
            Mask logits shaped ``(B, Q, H, W)`` where ``H * W = T``.
        """

        b, t, c = seg_tokens.shape
        h, w = self.patch_grid
        patch_map = seg_tokens.transpose(1, 2).reshape(b, c, h, w)
        return torch.einsum("bqc, bchw -> bqhw", self.head(queries), patch_map)
