"""Stage-2 multiview data scaffolding for 3D-aware DA3+EoMT training.

The goal is to mirror the multiview structure described in the integration
design while keeping the implementation lightweight enough to plug directly
into the existing EoMT Lightning training loop. The dataset yields RGB frames,
per-view intrinsics/poses, and 2D panoptic targets so the 2D segmentation loss
path stays identical to EoMT. Hooks for 3D ground truth are provided to enable
3D consistency losses without forcing a full ScanNet parser in-tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L


@dataclass
class MultiViewSample:
    """Container for a multiview training sample."""

    images: Tensor  # (V, C, H, W)
    intrinsics: Tensor  # (V, ...)
    poses: Tensor  # (V, ...)
    panoptic: List[Dict[str, Any]]  # EoMT-style 2D panoptic targets per view
    gt_3d: Optional[Tensor] = None


@dataclass
class MultiViewBatch:
    """Batch type for Stage-2 loaders.

    This mirrors the structure expected by the Stage-2 training loop: images
    are grouped by scene (batch dimension) and view, while panoptic targets are
    provided per-view so they can be flattened into the existing EoMT loss
    computation.
    """

    images: Tensor  # (B, V, C, H, W)
    intrinsics: Tensor  # (B, V, 3, 3)
    poses: Tensor  # (B, V, 4, 4)
    panoptic: List[List[Dict[str, Any]]]
    gt_3d: Optional[Tensor] = None


class ScanNetStage2Dataset(Dataset):
    """Toy multiview dataset placeholder.

    This keeps the interface compatible with the Stage-2 requirements while
    delegating real data decoding to downstream codebases. It returns
    zero-initialised tensors with the correct shapes so the trainer can be
    sanity-checked in CI.
    """

    def __init__(self, views_per_sample: int = 2, image_shape: Tuple[int, int] = (518, 518)):
        super().__init__()
        self.views_per_sample = views_per_sample
        self.image_shape = image_shape

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> MultiViewSample:
        h, w = self.image_shape
        images = torch.zeros(self.views_per_sample, 3, h, w)
        intrinsics = torch.eye(3).repeat(self.views_per_sample, 1, 1)
        poses = torch.eye(4).repeat(self.views_per_sample, 1, 1)
        # Minimal EoMT-style target scaffold per view
        panoptic: List[Dict[str, Any]] = []
        for _ in range(self.views_per_sample):
            panoptic.append(
                {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "masks": torch.zeros(0, h, w, dtype=torch.bool),
                    "is_crowd": torch.zeros(0, dtype=torch.bool),
                }
            )
        return MultiViewSample(
            images=images, intrinsics=intrinsics, poses=poses, panoptic=panoptic, gt_3d=None
        )


def build_stage2_dataloader(batch_size: int, num_workers: int, views_per_sample: int) -> DataLoader:
    dataset = ScanNetStage2Dataset(views_per_sample=views_per_sample)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_multiview,
    )


def collate_multiview(batch: Sequence[MultiViewSample]) -> MultiViewBatch:
    images = torch.stack([sample.images for sample in batch], dim=0)
    intrinsics = torch.stack([sample.intrinsics for sample in batch], dim=0)
    poses = torch.stack([sample.poses for sample in batch], dim=0)
    panoptic: List[List[Dict[str, Any]]] = [sample.panoptic for sample in batch]
    gt_3d = None
    for sample in batch:
        if sample.gt_3d is not None:
            gt_3d = sample.gt_3d if gt_3d is None else torch.cat([gt_3d, sample.gt_3d], dim=0)
    return MultiViewBatch(images=images, intrinsics=intrinsics, poses=poses, panoptic=panoptic, gt_3d=gt_3d)


class Stage2DataModule(L.LightningDataModule):
    """Lightning DataModule exposing multiview ScanNet-style data."""

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        views_per_sample: int = 2,
        image_shape: Tuple[int, int] = (518, 518),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.views_per_sample = views_per_sample
        self.image_shape = image_shape

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ScanNetStage2Dataset(
            views_per_sample=self.views_per_sample, image_shape=self.image_shape
        )
        self.val_dataset = ScanNetStage2Dataset(
            views_per_sample=self.views_per_sample, image_shape=self.image_shape
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_multiview,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_multiview,
        )

