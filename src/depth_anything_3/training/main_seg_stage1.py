from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch import cli
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
import torch

from depth_anything_3.training.coco_panoptic_datamodule import COCOPanopticDataModule
from depth_anything_3.training.seg_panoptic_module import DA3SegPanopticModule
from depth_anything_3.utils.checkpoint_utils import resolve_da3_ckpt_path


class SegLightningCLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("data.img_size", "model.img_size")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.stuff_classes", "model.stuff_classes", apply_on="instantiate")

        # 🔧 添加自定义参数
        parser.add_argument(
            "--resume_from_weights", 
            type=str, 
            default=None,
            help="Path to checkpoint for loading weights only (no optimizer state)"
        )

    def instantiate_classes(self) -> None:
        model_cfg = getattr(self.config, "model", None)
        if isinstance(model_cfg, DictConfig):
            raw_path = model_cfg.get("da3_pretrained_path", "") or ""
            resolved = resolve_da3_ckpt_path(raw_path) if raw_path else ""
            if resolved and resolved != raw_path:
                rank_zero_info(
                    "Resolved DA3 pretrained path for segmentation stage1: %s -> %s",
                    raw_path,
                    resolved,
                )
                model_cfg.da3_pretrained_path = resolved

        super().instantiate_classes()

    def before_fit(self):
        """在fit之前加载权重"""
        # 🔧 正确获取参数：检查多个可能的位置
        resume_path = None
        
        # 尝试从不同位置获取
        if hasattr(self.config, "resume_from_weights"):
            resume_path = self.config.resume_from_weights
        elif hasattr(self.config, "fit") and hasattr(self.config.fit, "resume_from_weights"):
            resume_path = self.config.fit.resume_from_weights
        
        if resume_path and resume_path not in ("", "null", None):
            import torch
            rank_zero_info(f"🔄 Loading weights from {resume_path} (weights only, no optimizer state)")
            
            try:
                checkpoint = torch.load(resume_path, map_location="cpu")
                if "state_dict" in checkpoint:
                    # 只加载模型权重
                    missing, unexpected = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
                    rank_zero_info(f"✅ Weights loaded successfully")
                    if missing:
                        rank_zero_info(f"   Missing keys: {len(missing)} (new parameters)")
                    if unexpected:
                        rank_zero_info(f"   Unexpected keys: {len(unexpected)} (removed parameters)")
                else:
                    rank_zero_info("❌ No 'state_dict' found in checkpoint")
            except Exception as e:
                rank_zero_info(f"❌ Failed to load weights: {e}")
        else:
            rank_zero_info("ℹ️  No resume_from_weights specified, starting from scratch (with DA3 pretrained)")

    # def fit(self, model: pl.LightningModule, datamodule: pl.LightningDataModule, **kwargs):
    #     ckpt_path = kwargs.pop("ckpt_path", None)
    #     if ckpt_path is None:
    #         ckpt_path = getattr(self.config, "ckpt_path", None)
    #     if ckpt_path in ("", "null"):
    #         ckpt_path = None

    #     return super().fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path, **kwargs)


def cli_main():
    torch.set_float32_matmul_precision("medium")
    
    SegLightningCLI(
        DA3SegPanopticModule,
        COCOPanopticDataModule,
        # save_config_overwrite=True,
        trainer_defaults={
            "accelerator": "gpu",
            "devices": 8,
            "strategy": "ddp",
            "precision": "16-mixed",
            "gradient_clip_val": 0.01,
            "enable_model_summary": False,
            "callbacks": [ModelSummary(max_depth=3), LearningRateMonitor(logging_interval="step", log_momentum=False,)],
        },
    )


if __name__ == "__main__":
    cli_main()
