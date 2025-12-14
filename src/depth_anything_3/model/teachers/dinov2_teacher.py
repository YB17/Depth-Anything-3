from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from lightning.pytorch.utilities import rank_zero_info

from depth_anything_3.model.dinov2.vision_transformer import vit_base, vit_large


DINO_V2_MEAN = (0.485, 0.456, 0.406)
DINO_V2_STD = (0.229, 0.224, 0.225)


_DINOV2_URLS = {
    "vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14_pretrain.pth",
    "vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14_pretrain.pth",
}


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 teacher that returns patch tokens for selected layers."""

    def __init__(
        self,
        variant: str = "vitb14",
        layers: Iterable[int] | None = None,
        input_size: Tuple[int, int] = (518, 518),
        use_teacher_normalize: bool = True,
        pretrained_path: str = "",
        allow_missing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = list(layers or [])
        self.input_size = tuple(input_size)
        self.use_teacher_normalize = use_teacher_normalize
        self.allow_missing = allow_missing

        if variant not in {"vitb14", "vitl14"}:
            raise ValueError(f"Unsupported DINOv2 variant: {variant}")
        if variant == "vitb14":
            self.model = vit_base(img_size=self.input_size[0], patch_size=14, cat_token=False)
        else:
            self.model = vit_large(img_size=self.input_size[0], patch_size=14, cat_token=False)

        self.patch_size = self.model.patch_embed.patch_size
        self.patch_grid = self.model.patch_embed.patches_resolution

        self._load_weights(pretrained_path or _DINOV2_URLS.get(variant, ""))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        rank_zero_info(
            "[DINOv2Teacher] variant=%s patch=%s img_size=%s grid=%s layers=%s",
            variant,
            self.patch_size,
            self.input_size,
            self.patch_grid,
            self.layers,
        )

    def _load_weights(self, path_or_url: str) -> None:
        if not path_or_url:
            raise ValueError("A pretrained checkpoint path or URL must be provided for the teacher.")
        if Path(path_or_url).is_file():
            state_dict = torch.load(path_or_url, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu", check_hash=False)

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
            cleaned_state[new_key] = v

        missing, unexpected = self.model.load_state_dict(cleaned_state, strict=not self.allow_missing)
        if missing or unexpected:
            rank_zero_info(
                "[DINOv2Teacher] load_state strict=%s missing=%d unexpected=%d",
                not self.allow_missing,
                len(missing),
                len(unexpected),
            )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        if images.dim() > 4:
            images = images.squeeze(1)
        if self.use_teacher_normalize:
            mean = torch.tensor(DINO_V2_MEAN, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
            std = torch.tensor(DINO_V2_STD, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
            images = (images - mean) / std

        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)

        outputs, _, _ = self.model.get_intermediate_layers(
            images,
            n=self.layers,
            seg_attn_mask_fn=None,
            seg_head_fn=None,
            apply_seg_head_to_intermediate=False,
            apply_seg_head_to_last=False,
        )
        token_dict: Dict[int, torch.Tensor] = {}
        for idx, layer_idx in enumerate(self.layers):
            # outputs[idx] is a tuple (patch_tokens, camera_tokens)
            token_dict[layer_idx] = outputs[idx][0]
        return token_dict
