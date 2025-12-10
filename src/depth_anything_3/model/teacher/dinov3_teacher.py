from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from third_party.dinov3.models.vision_transformer import vit_base


class DINOv3Teacher(nn.Module):
    """Wrapper for frozen DINOv3-base teacher with intermediate patch tokens."""

    def __init__(
        self,
        ckpt_path: str,
        img_size: int = 592,
        layers: Iterable[int] | None = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.layers = list(layers or [5, 7, 9, 11])
        self.model = vit_base(img_size=img_size, patch_size=patch_size)
        self._load_weights(ckpt_path)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_weights(self, ckpt_path: str) -> None:
        if not ckpt_path:
            return
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ["state_dict", "model", "teacher", "module"]:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        cleaned_state = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone.") :]
            cleaned_state[new_key] = v
        missing, unexpected = self.model.load_state_dict(cleaned_state, strict=False)
        if missing:
            print(f"[DINOv3Teacher] Missing keys: {missing}")
        if unexpected:
            print(f"[DINOv3Teacher] Unexpected keys: {unexpected}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        outputs = self.model.get_intermediate_layers(
            images,
            n=self.layers,
            return_class_token=False,
            return_extra_tokens=False,
            reshape=False,
            norm=True,
        )
        token_dict: Dict[int, torch.Tensor] = {}
        for idx, layer_idx in enumerate(self.layers):
            token_dict[layer_idx] = outputs[idx]
        return token_dict
