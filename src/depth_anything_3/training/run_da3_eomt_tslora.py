"""CLI entry to train DA3+EoMT with token-selective LoRA using Lightning."""

from __future__ import annotations

import argparse
import importlib
import yaml

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from depth_anything_3.data.scannet_stage2 import Stage2DataModule
from depth_anything_3.training.da3_eomt_tslora_module import (
    DA3EoMTTSLoRALightning,
)
from depth_anything_3.utils.checkpoint_utils import (
    load_da3_pretrained_backbone,
    resolve_da3_ckpt_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA3+EoMT TS-LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--wandb_project", type=str, default="da3-eomt-tslora", help="Wandb project"
    )
    parser.add_argument(
        "--wandb_run", type=str, default=None, help="Optional Wandb run name"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})

    builder_path = model_cfg.get("builder")
    if builder_path is None:
        raise ValueError("model.builder must point to a callable builder")

    module_name, func_name = builder_path.rsplit(":", 1)
    builder = getattr(importlib.import_module(module_name), func_name)
    num_classes = trainer_cfg.get("num_classes")
    network = builder(model_cfg, num_classes=num_classes)
    ckpt_path = resolve_da3_ckpt_path(model_cfg.get("da3_pretrained_path", "") or "")
    if ckpt_path and hasattr(network, "backbone"):
        load_da3_pretrained_backbone(network.backbone, ckpt_path, strict=False)

    trainer_cfg.setdefault("lambda_2d", model_cfg.get("lambda_2d", 1.0))
    trainer_cfg.setdefault("lambda_3d", model_cfg.get("lambda_3d", 0.0))
    lightning_module = DA3EoMTTSLoRALightning(network=network, **trainer_cfg)

    logger = WandbLogger(project=args.wandb_project, name=args.wandb_run)
    trainer = L.Trainer(logger=logger)

    if data_cfg.get("stage", "2d") == "3d":
        datamodule = Stage2DataModule(
            batch_size=data_cfg.get("batch_size", 1),
            num_workers=data_cfg.get("num_workers", 0),
            views_per_sample=data_cfg.get("views_per_sample", 2),
            image_shape=tuple(data_cfg.get("image_shape", (518, 518))),
        )
        trainer.fit(lightning_module, datamodule=datamodule)
    else:
        datamodule = Stage2DataModule(
            batch_size=data_cfg.get("batch_size", 1),
            num_workers=data_cfg.get("num_workers", 0),
            views_per_sample=data_cfg.get("views_per_sample", 1),
            image_shape=tuple(data_cfg.get("image_shape", (518, 518))),
        )
        trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()

