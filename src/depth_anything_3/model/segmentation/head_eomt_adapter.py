from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from third_party.eomt.models.scale_block import ScaleBlock


class EoMTSegHead(nn.Module):
    """Adapter that reuses EoMT-style heads for DA3 segmentation tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_classes: int,
        patch_grid: Tuple[int, int],
        num_upscale_blocks: int | None = None,
        hidden_mult: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.patch_grid = patch_grid

        self.query_norm = nn.LayerNorm(embed_dim)
        self.class_head = nn.Linear(embed_dim, hidden_mult * embed_dim)
        self.logit_head = nn.Linear(hidden_mult * embed_dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if num_upscale_blocks is None:
            max_patch = max(patch_grid[0], patch_grid[1])
            num_upscale_blocks = max(1, int(torch.tensor(max_patch).log2().item()) - 2)
        self.upscale = nn.Sequential(*[ScaleBlock(embed_dim) for _ in range(num_upscale_blocks)])

    def forward(self, g_seg: Tensor, queries: Tensor) -> Dict[str, Tensor]:
        """Predict masks and logits from segmentation tokens.

        Args:
            g_seg: Segmentation geometry tokens shaped ``(B, T, C)``.
            queries: Segmentation query tokens shaped ``(B, Q, C)``.

        Returns:
            Dict with ``pred_masks`` and ``pred_logits`` mirroring EoMT outputs.
        """

        queries = self.query_norm(queries)
        class_feats = self.class_head(queries)
        pred_logits = self.logit_head(class_feats)

        patch_map = g_seg.transpose(1, 2).reshape(g_seg.shape[0], -1, *self.patch_grid)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(queries), self.upscale(patch_map))

        return {"pred_masks": mask_logits, "pred_logits": pred_logits}
