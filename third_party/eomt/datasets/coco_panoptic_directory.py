# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
# from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset, SequentialSampler
import torch

# from datasets.lightning_data_module import LightningDataModule
# from datasets.directory_dataset import DirectoryDataset
# from datasets.transforms import Transforms, MinimalTransforms
from .lightning_data_module import LightningDataModule
from .directory_dataset import DirectoryDataset
from .transforms import Transforms

# MinimalTransforms 可能不存在
try:
    from .transforms import MinimalTransforms
except (ImportError, AttributeError):
    MinimalTransforms = Transforms
    
CLASS_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    92: 80,
    93: 81,
    95: 82,
    100: 83,
    107: 84,
    109: 85,
    112: 86,
    118: 87,
    119: 88,
    122: 89,
    125: 90,
    128: 91,
    130: 92,
    133: 93,
    138: 94,
    141: 95,
    144: 96,
    145: 97,
    147: 98,
    148: 99,
    149: 100,
    151: 101,
    154: 102,
    155: 103,
    156: 104,
    159: 105,
    161: 106,
    166: 107,
    168: 108,
    171: 109,
    175: 110,
    176: 111,
    177: 112,
    178: 113,
    180: 114,
    181: 115,
    184: 116,
    185: 117,
    186: 118,
    187: 119,
    188: 120,
    189: 121,
    190: 122,
    191: 123,
    192: 124,
    193: 125,
    194: 126,
    195: 127,
    196: 128,
    197: 129,
    198: 130,
    199: 131,
    200: 132,
}


class COCOPanopticDirectory(LightningDataModule):
    """
    使用DirectoryDataset的COCO全景分割数据集
    适用于已解压的数据集文件
    
    数据集目录结构应该如下：
    coco_data/
    ├── train2017/           # 训练图像
    ├── val2017/             # 验证图像
    ├── panoptic_train2017/  # 训练全景分割标注
    ├── panoptic_val2017/    # 验证全景分割标注
    └── annotations/         # JSON标注文件
        ├── panoptic_train2017.json
        └── panoptic_val2017.json
    """
    def __init__(
        self,
        path,
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 133,
        color_jitter_enabled: bool = False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transform_cfg: dict | None = None,
        val_transform_cfg: dict | None = None,
        resolution_preset: str = "Safe",
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.stuff_classes = stuff_classes

        self.resolution_preset = resolution_preset
        self.train_transform_cfg = train_transform_cfg or {}
        self.val_transform_cfg = val_transform_cfg or {}

        self.train_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=self.train_transform_cfg.get("color_jitter", color_jitter_enabled),
            scale_range=(1.0, 1.0),
            short_edge_range=tuple(self.train_transform_cfg.get("short_edge", [])) if self.train_transform_cfg.get("short_edge") else None,
            max_long_edge=self.train_transform_cfg.get("long_edge_max"),
            crop_sizes=self.train_transform_cfg.get("crop_sizes"),
            keep_full_instances=self.train_transform_cfg.get("keep_full_instances", True),
            flip_prob=float(self.train_transform_cfg.get("flip_prob", 0.5)),
        )
        self.eval_transforms = MinimalTransforms(img_size=img_size)
        self.tiny_train_size = 32
        self.tiny_val_size = 32


    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, is_thing_by_id=None, **kwargs):
        """
        解析COCO全景分割目标
        COCO全景分割使用RGB编码：segment_id = R + G*256 + B*256^2
        """
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2

        masks, labels, is_crowd, is_thing, target_ids = [], [], [], [], []

        for label_id in target.unique():
            lid = label_id.item()
        
            if lid not in labels_by_id:
                continue

            cls_id = labels_by_id[lid]
            if cls_id not in CLASS_MAPPING:
                continue

            target_ids.append(lid)
            masks.append(target == label_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[lid])
            if is_thing_by_id is not None:
                is_thing.append(is_thing_by_id.get(lid, False))
            else:
                is_thing.append(False)

        # 如果没有找到有效的masks，创建一个背景mask
        if not masks:
            h, w = target.shape[-2:]
            masks = [torch.zeros((h, w), dtype=torch.bool)]
            labels = [0]  # 背景类别
            is_crowd = [False]
            is_thing = [False]
            target_ids = [0]  # 🔧 添加缺失的target_ids

        return masks, labels, is_crowd, is_thing, target_ids

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        if stage == "fit" or stage is None:
            self.train_dataset = DirectoryDataset(
                data_path=Path(self.path),
                img_suffix=".jpg",
                target_parser=self.target_parser,
                transforms=self.train_transforms,
                only_annotations_json=False,
                target_suffix=".png",
                stuff_classes=self.stuff_classes,
                img_folder_path=Path("train2017"),
                target_folder_path=Path("panoptic_train2017"),
                annotations_json_path=Path("annotations/panoptic_train2017.json"),
                check_empty_targets=self.check_empty_targets,
                class_mapping=CLASS_MAPPING,
            )

            self.val_dataset = DirectoryDataset(
                data_path=Path(self.path),
                img_suffix=".jpg",
                target_parser=self.target_parser,
                transforms=self.eval_transforms,
                only_annotations_json=False,
                target_suffix=".png",
                stuff_classes=self.stuff_classes,
                img_folder_path=Path("val2017"),
                target_folder_path=Path("panoptic_val2017"),
                annotations_json_path=Path("annotations/panoptic_val2017.json"),
                check_empty_targets=self.check_empty_targets,
                class_mapping=CLASS_MAPPING,
            )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            # self.train_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        ) 