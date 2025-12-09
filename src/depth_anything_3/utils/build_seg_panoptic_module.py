from __future__ import annotations

from typing import Any, Tuple

from omegaconf import DictConfig, OmegaConf

from depth_anything_3.training.coco_panoptic_datamodule import COCOPanopticDataModule
from depth_anything_3.training.seg_panoptic_module import DA3SegPanopticModule
from depth_anything_3.utils.checkpoint_utils import resolve_da3_ckpt_path


def _as_tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        return tuple(value)  # type: ignore[arg-type]
    return (value, value)


def build_model_and_datamodule_from_config(cfg: DictConfig) -> tuple[DA3SegPanopticModule, COCOPanopticDataModule]:
    """Instantiate model and datamodule using a Hydra/OmegaConf config.

    This mirrors the training entry so inference scripts can reuse the same
    construction logic without duplicating CLI code.
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    model_cfg = dict(cfg_dict.get("model", {}) or {})
    data_cfg = dict(cfg_dict.get("data", {}) or {})

    img_size = _as_tuple(model_cfg.get("img_size") or data_cfg.get("img_size") or (640, 640))
    num_classes = model_cfg.get("num_classes", data_cfg.get("num_classes", 133))
    stuff_classes = model_cfg.get("stuff_classes", data_cfg.get("stuff_classes", []))

    datamodule = COCOPanopticDataModule(
        root=data_cfg.get("root", ""),
        panoptic_json_train=data_cfg.get("panoptic_json_train", ""),
        panoptic_json_val=data_cfg.get("panoptic_json_val", ""),
        stuff_classes=stuff_classes,
        img_size=_as_tuple(data_cfg.get("img_size", img_size)),
        num_classes=num_classes,
        batch_size_per_gpu=data_cfg.get("batch_size_per_gpu", data_cfg.get("batch_size", 4)),
        num_workers=data_cfg.get("num_workers", 8),
        color_jitter_enabled=data_cfg.get("color_jitter_enabled", True),
        scale_range=_as_tuple(data_cfg.get("scale_range", (0.1, 2.0))),
        check_empty_targets=data_cfg.get("check_empty_targets", True),
    )

    raw_pretrained = model_cfg.get("da3_pretrained_path", "")
    if raw_pretrained:
        resolved = resolve_da3_ckpt_path(raw_pretrained)
        if resolved:
            model_cfg["da3_pretrained_path"] = resolved

    model_cfg.setdefault("img_size", img_size)
    model_cfg.setdefault("num_classes", num_classes)
    model_cfg.setdefault("stuff_classes", stuff_classes)

    model = DA3SegPanopticModule(**model_cfg)

    return model, datamodule
