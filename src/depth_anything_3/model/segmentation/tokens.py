from __future__ import annotations

import torch
from torch import Tensor, nn


class SegmentationTokens(nn.Module):
    """Initialize segmentation branch tokens (B, G_seg, S).

    The class mirrors the design outlined in the segmentation branch proposal:
    - ``B``: bottleneck tokens compressing geometry.
    - ``G_seg``: segmentation geometry tokens cloned from backbone geometry.
    - ``S``: learnable segmentation queries.
    """

    def __init__(self, embed_dim: int, num_bottleneck_tokens: int, num_queries: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.num_queries = num_queries

        self.bottleneck = nn.Parameter(torch.zeros(1, num_bottleneck_tokens, embed_dim))
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, geom_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create initial states for the segmentation branch.

        Args:
            geom_tokens: Geometry tokens ``(B, T, C)`` from the backbone.

        Returns:
            Tuple ``(B_0, G_seg_0, S_0)`` matching the batch/device/dtype of
            ``geom_tokens``.
        """

        batch = geom_tokens.shape[0]
        B0 = self.bottleneck.expand(batch, -1, -1).to(device=geom_tokens.device, dtype=geom_tokens.dtype)
        G_seg0 = geom_tokens.detach().clone()
        S0 = self.queries.expand(batch, -1, -1).to(device=geom_tokens.device, dtype=geom_tokens.dtype)
        return B0, G_seg0, S0
