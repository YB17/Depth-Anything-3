# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import re
import json
import glob
from pathlib import Path
from typing import Callable, Optional
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# 移除循环导入
# from .coco_panoptic_directory import CLASS_MAPPING


class DirectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_data_path: Optional[Path] = None,
        target_instance_data_path: Optional[Path] = None,
        img_folder_path: Path = Path("./"),
        target_folder_path: Path = Path("./"),
        target_instance_folder_path: Path = Path("./"),
        annotations_json_path: Optional[Path] = None,
        class_mapping: Optional[dict] = None,  # 新增参数
    ):
        """
        DirectoryDataset - 从文件系统目录直接读取数据的数据集类
        
        参数说明:
            data_path: 数据根目录路径
            img_suffix: 图像文件后缀 (如 ".jpg")
            target_parser: 目标解析函数
            check_empty_targets: 是否检查空目标
            transforms: 数据变换函数
            only_annotations_json: 是否只使用JSON标注文件
            target_suffix: 目标文件后缀 (如 ".png")
            stuff_classes: stuff类别列表
            img_stem_suffix: 图像文件名后缀
            target_stem_suffix: 目标文件名后缀
            target_data_path: 目标数据路径（如果与图像路径不同）
            target_instance_data_path: 实例目标数据路径
            img_folder_path: 图像文件夹相对路径
            target_folder_path: 目标文件夹相对路径
            target_instance_folder_path: 实例目标文件夹相对路径
            annotations_json_path: JSON标注文件路径
            class_mapping: 类别映射字典
        """
        self.data_path = data_path
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_data_path = target_data_path or data_path
        self.target_instance_data_path = target_instance_data_path
        
        # 构建完整路径
        self.img_folder_full_path = data_path / img_folder_path
        self.target_folder_full_path = self.target_data_path / target_folder_path
        self.target_instance_folder_full_path = (
            self.target_instance_data_path / target_instance_folder_path
            if target_instance_data_path else None
        )

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.is_thing_by_id = {}

        self.class_mapping = class_mapping or {}

        # 处理JSON标注文件
        if annotations_json_path is not None:
            json_file_path = self.target_data_path / annotations_json_path
            with open(json_file_path, 'r') as file:
                annotation_data = json.load(file)

            # image_id_to_file_name = {
            #     image["id"]: image["file_name"] for image in annotation_data["images"]
            # }

            # 新增：保存 image_id_to_file_name 字典到文件
            # save_path = self.target_data_path / "image_id_to_file_name.json"
            # print(f"save_path: {save_path}")
            # with open(save_path, "w") as f:
            #     json.dump(image_id_to_file_name, f, indent=2, ensure_ascii=False)

            for annotation in annotation_data["annotations"]:
                # print(f"type of annotation['image_id']: {type(annotation['image_id'])}")
                # img_filename = annotation[(annotation["image_id"])]
                img_filename = annotation["file_name"].replace(target_suffix, img_suffix)

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_thing_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info.get("isthing", 0))
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        # 获取所有图像文件
        img_pattern = f"**/*{img_stem_suffix}{img_suffix}"
        img_files = list(self.img_folder_full_path.glob(img_pattern))
        
        for img_file in sorted(img_files, key=self._sort_key_path):
            if not img_file.is_file():
                print(f"img_file: {img_file} is not a file")
                continue

            img_relative_path = img_file.relative_to(self.img_folder_full_path)
            
            if not only_annotations_json:
                # 构建对应的目标文件路径
                target_parent = self.target_folder_full_path / img_relative_path.parent
                target_stem = img_relative_path.stem.replace(img_stem_suffix, target_stem_suffix)
                target_file_path = target_parent / f"{target_stem}{target_suffix}"

            # 检查标注数据
            if self.labels_by_id:
                if img_file.name not in self.labels_by_id:
                    continue
                if not self.labels_by_id[img_file.name]:
                    continue
                
                # 新增：检查是否有有效的类别映射
                if check_empty_targets and self.class_mapping:
                    valid_classes = [
                        cls_id for cls_id in self.labels_by_id[img_file.name].values()
                        if cls_id in self.class_mapping  # 使用传入的class_mapping
                    ]
                    if not valid_classes:
                        print(f"跳过图像 {img_file.name}：没有有效的类别映射")
                        continue
            else:
                if not only_annotations_json:
                    if not target_file_path.exists():
                        continue

                if check_empty_targets:
                    min_val, max_val = Image.open(target_file_path).getextrema()
                    if min_val == max_val:
                        continue

            # 检查实例目标文件
            if self.target_instance_folder_full_path is not None:
                target_instance_parent = self.target_instance_folder_full_path / img_relative_path.parent
                target_instance_file_path = target_instance_parent / f"{target_stem}{target_suffix}"

                if check_empty_targets and target_instance_file_path.exists():
                    extrema = Image.open(target_instance_file_path).getextrema()
                    if all(min_val == max_val for min_val, max_val in extrema):
                        if not only_annotations_json:
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(Image.open(target_file_path)),
                                target_instance=tv_tensors.Mask(Image.open(target_instance_file_path)),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                print(f"img_file.name: {img_file.name} target_file_path: {target_file_path} is empty")
                                continue

            # 添加到列表
            self.imgs.append(str(img_file))

            if not only_annotations_json:
                self.targets.append(str(target_file_path))

            if self.target_instance_folder_full_path is not None:
                self.targets_instance.append(str(target_instance_file_path))

        print(f"共找到 {len(self.imgs)} 张图片用于训练")
        if len(self.imgs) < 10:
            print("训练图片样本：", self.imgs)

    def __getitem__(self, index: int):
        # 加载图像
        img_path = Path(self.imgs[index])
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        # 🔧 提取图像ID
        filename = img_path.stem
        # 对于COCO格式，提取数字部分作为图像ID
        import re
        match = re.search(r'(\d+)', filename)
        if match:
            # 去除前导零，但保持至少一位数字
            image_id = match.group(1).lstrip('0') or '0'
            # 如果原始长度是12位，说明是COCO标准格式，保持完整格式
            if len(match.group(1)) == 12:
                image_id = match.group(1)
        else:
            image_id = filename

        # 加载目标
        target = None
        if not self.only_annotations_json:
            target_path = Path(self.targets[index])
            target = tv_tensors.Mask(Image.open(target_path), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        # 加载实例目标
        target_instance = None
        if self.targets_instance:
            target_instance_path = Path(self.targets_instance[index])
            if target_instance_path.exists():
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance_path), dtype=torch.long
                )

        # 解析目标
        masks, labels, is_crowd, is_thing, target_ids = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(img_path.name, {}),
            labels_by_id=self.labels_by_id.get(img_path.name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(img_path.name, {}),
            is_thing_by_id=getattr(self, "is_thing_by_id", {}).get(img_path.name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
            "is_thing": torch.tensor(is_thing),
            "target_ids": torch.tensor(target_ids),
            "image_id": image_id,  # 🔧 添加图像ID到target字典中
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def _sort_key_path(path: Path):
        """为文件路径排序的键函数"""
        match = re.search(r"\d+", path.name)
        return (int(match.group()) if match else float("inf"), str(path))

    def __len__(self):
        return len(self.imgs)

    def close(self):
        """为了与原Dataset类保持接口一致，但目录数据集不需要关闭操作"""
        pass

    def __del__(self):
        self.close()

    def __getstate__(self):
        """支持pickle序列化"""
        return self.__dict__.copy() 