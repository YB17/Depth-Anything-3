# Minimal DINOv3 model registry for teacher loading
from third_party.dinov3.models import vision_transformer as vits

__all__ = ["vits", "build_vit_base"]

def build_vit_base(img_size: int = 592, patch_size: int = 16):
    return vits.vit_base(img_size=img_size, patch_size=patch_size)
