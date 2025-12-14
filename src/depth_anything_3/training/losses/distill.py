from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from third_party.dinov3.loss.gram_loss import GramLoss

LossType = Literal["gram", "cosine", "l2"]


class DistillLoss:
    """Utility to compute different distillation objectives on patch tokens."""

    def __init__(self) -> None:
        self.gram_loss = GramLoss()

    def compute(
        self,
        student_patch: torch.Tensor,
        teacher_patch: torch.Tensor,
        loss_type: LossType = "cosine",
        normalize: bool = True,
        fp32: bool = True,
    ) -> torch.Tensor:
        if fp32:
            student_patch = student_patch.float()
            teacher_patch = teacher_patch.float()

        if loss_type == "gram":
            return self.gram_loss(student_patch, teacher_patch)

        if normalize:
            student_patch = F.normalize(student_patch, dim=-1)
            teacher_patch = F.normalize(teacher_patch, dim=-1)

        if loss_type == "cosine":
            loss = 1.0 - (student_patch * teacher_patch).sum(dim=-1)
            return loss.mean()
        if loss_type == "l2":
            return F.mse_loss(student_patch, teacher_patch)

        raise ValueError(f"Unsupported distillation loss type: {loss_type}")
