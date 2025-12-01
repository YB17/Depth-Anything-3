"""Token-selective LoRA modules.

This module provides linear layers that only apply the LoRA residual to
user-specified tokens. It is designed to keep the base model behaviour intact
for geometry tokens while adapting segmentation query tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class TokenSelectiveLoRAConfig:
    """Configuration for :class:`TokenSelectiveLoRALinear`.

    Attributes:
        rank: LoRA rank ``r``.
        alpha: Scaling factor applied to the LoRA update.
        enable: Whether LoRA is active.
    """

    rank: int = 4
    alpha: float = 1.0
    enable: bool = False


class TokenSelectiveLoRALinear(nn.Module):
    """Linear layer with token-selective LoRA residuals."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, enable: bool = True):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.enable_lora = enable and rank > 0

        if self.enable_lora:
            self.lora_A = nn.Parameter(torch.zeros((rank, base.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        """Compute the base projection plus optional token-gated LoRA residual.

        Args:
            x: Input tensor of shape ``[B, N, C]`` or ``[B*N, C]``.
            token_mask: Optional mask indicating which tokens should receive the
                LoRA residual. Supported shapes are ``[B, N]`` or ``[B*N]``; the
                mask is broadcast over the feature dimension.
        """

        base_out = self.base(x)
        if not self.enable_lora or self.lora_A is None or self.lora_B is None:
            return base_out

        if token_mask is None:
            token_mask = torch.ones_like(base_out[..., :1], dtype=base_out.dtype)
        else:
            if token_mask.dim() == base_out.dim() - 1:
                token_mask = token_mask.unsqueeze(-1)
            token_mask = token_mask.to(base_out.dtype)

        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        scaling = self.alpha / float(self.rank)
        delta = delta * scaling
        delta = delta * token_mask
        return base_out + delta

    def set_lora_enabled(self, flag: bool) -> None:
        """Enable or disable the LoRA residual at runtime.

        If re-enabled after being disabled, LoRA parameters are (re)created and
        initialised so the residual path becomes trainable again.
        """

        if flag and (self.lora_A is None or self.lora_B is None):
            self.lora_A = nn.Parameter(torch.zeros((self.rank, self.base.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.base.out_features, self.rank)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        self.enable_lora = flag and self.rank > 0


def wrap_linear_with_tslora(
    linear: nn.Linear, rank: int, alpha: float, enable: bool = True
) -> TokenSelectiveLoRALinear:
    """Wrap an ``nn.Linear`` with :class:`TokenSelectiveLoRALinear`."""

    wrapped = TokenSelectiveLoRALinear(linear, rank=rank, alpha=alpha, enable=enable)
    wrapped.base.weight = linear.weight
    wrapped.base.bias = linear.bias
    return wrapped
