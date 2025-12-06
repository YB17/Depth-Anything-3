from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch import cli
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import WandbLogger
import torch

from depth_anything_3.training.coco_panoptic_datamodule import COCOPanopticDataModule
from depth_anything_3.training.seg_panoptic_module import DA3SegPanopticModule


class SegLightningCLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.num_classes", "model.init_args.num_classes", apply_on="instantiate")
        parser.link_arguments("data.init_args.stuff_classes", "model.init_args.stuff_classes", apply_on="instantiate")

    def fit(self, model: pl.LightningModule, datamodule: pl.LightningDataModule, **kwargs):
        logger = kwargs.get("trainer", {}).get("logger", None)
        if isinstance(logger, WandbLogger):
            logger.log_code(".")
        torch.set_float32_matmul_precision("medium")
        return super().fit(model=model, datamodule=datamodule, **kwargs)


def cli_main():
    SegLightningCLI(
        DA3SegPanopticModule,
        COCOPanopticDataModule,
        save_config_overwrite=True,
        trainer_defaults={
            "accelerator": "gpu",
            "devices": 8,
            "strategy": "ddp",
            "precision": "16-mixed",
            "gradient_clip_val": 0.01,
            "enable_model_summary": False,
            "callbacks": [ModelSummary(max_depth=3), LearningRateMonitor(logging_interval="epoch")],
        },
    )


if __name__ == "__main__":
    cli_main()
