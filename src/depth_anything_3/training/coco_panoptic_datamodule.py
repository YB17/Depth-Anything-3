from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from third_party.eomt.datasets.coco_panoptic_directory import COCOPanopticDirectory, CLASS_MAPPING


class COCOPanopticDataModule(COCOPanopticDirectory):
    """Wrapper for COCOPanopticDirectory with adapted parameters."""

    def __init__(
        self,
        root: str,
        panoptic_json_train: str,
        panoptic_json_val: str,
        stuff_classes: list[int],
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 133,
        batch_size_per_gpu: int = 4,
        num_workers: int = 8,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True,
    ) -> None:
        # 将参数适配到 COCOPanopticDirectory 的接口
        super().__init__(
            path=root,
            stuff_classes=stuff_classes,
            num_workers=num_workers,
            batch_size=batch_size_per_gpu,
            img_size=img_size,
            num_classes=num_classes,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            check_empty_targets=check_empty_targets,
        )

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        # 解码RGB编码的segment_id
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2

        masks, labels, is_crowd, is_thing, target_ids = [], [], [], [], []

        for label_id in target.unique():
            lid = label_id.item()
            if lid not in labels_by_id:
                continue

            cls_id = labels_by_id[lid]
            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target == label_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[lid])
            # 添加 is_thing 和 target_ids
            is_thing.append(cls_id < 80)  # COCO中，类别ID < 80 的是thing类
            target_ids.append(lid)

        return masks, labels, is_crowd, is_thing, target_ids

    @staticmethod
    def train_collate(batch):
        """重写collate函数，添加视图维度 (B, 3, H, W) -> (B, 1, 3, H, W)"""
        imgs, targets = [], []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        # 堆叠并添加视图维度 N=1
        imgs_stacked = torch.stack(imgs).unsqueeze(1)  # (B, 3, H, W) -> (B, 1, 3, H, W)
        # 🔧 关键修复：将uint8转换为float32，并归一化到[0, 1]
        if imgs_stacked.dtype == torch.uint8:
            imgs_stacked = imgs_stacked.float() / 255.0
        return imgs_stacked, targets

    @staticmethod
    def eval_collate(batch):
        """重写collate函数，添加视图维度"""
        imgs, targets = zip(*batch)
        # 堆叠并添加视图维度
        imgs_stacked = torch.stack(imgs).unsqueeze(1)  # (B, 3, H, W) -> (B, 1, 3, H, W)
        # 🔧 关键修复：将uint8转换为float32，并归一化到[0, 1]
        if imgs_stacked.dtype == torch.uint8:
            imgs_stacked = imgs_stacked.float() / 255.0

        return imgs_stacked, list(targets)