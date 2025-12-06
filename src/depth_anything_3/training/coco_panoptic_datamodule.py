from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from third_party.eomt.datasets.coco_panoptic import CLASS_MAPPING
from third_party.eomt.datasets.dataset import Dataset
from third_party.eomt.datasets.transforms import Transforms


class COCOPanopticDataModule(pl.LightningDataModule):
    """Panoptic COCO datamodule aligned with the EoMT dataset API."""

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
        super().__init__()
        self.root = Path(root)
        self.panoptic_json_train = Path(panoptic_json_train)
        self.panoptic_json_val = Path(panoptic_json_val)
        self.stuff_classes = stuff_classes
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = num_workers
        self.color_jitter_enabled = color_jitter_enabled
        self.scale_range = scale_range
        self.check_empty_targets = check_empty_targets

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2

        masks, labels, is_crowd = [], [], []

        for label_id in target.unique():
            if label_id.item() not in labels_by_id:
                continue

            cls_id = labels_by_id[label_id.item()]
            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target == label_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id.item()])

        return masks, labels, is_crowd

    def setup(self, stage: str | None = None) -> None:
        transforms = Transforms(
            img_size=self.img_size,
            color_jitter_enabled=self.color_jitter_enabled,
            scale_range=self.scale_range,
        )

        dataset_kwargs: dict[str, Any] = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }

        self.train_dataset = Dataset(
            transforms=transforms,
            img_folder_path_in_zip=Path(self.root, "train2017"),
            target_folder_path_in_zip=Path(self.root, "panoptic_train2017"),
            annotations_json_path_in_zip=self.panoptic_json_train,
            target_zip_path_in_zip=self.panoptic_json_train.parent,
            target_zip_path=self.panoptic_json_train,
            zip_path=Path(self.root, "train2017"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            transforms=transforms,
            img_folder_path_in_zip=Path(self.root, "val2017"),
            target_folder_path_in_zip=Path(self.root, "panoptic_val2017"),
            annotations_json_path_in_zip=self.panoptic_json_val,
            target_zip_path_in_zip=self.panoptic_json_val.parent,
            target_zip_path=self.panoptic_json_val,
            zip_path=Path(self.root, "val2017"),
            **dataset_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.train_dataset.train_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.val_dataset.eval_collate,
        )

