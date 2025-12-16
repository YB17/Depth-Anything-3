"""Stage-1 training entrypoint for DA3 + EoMT + token-selective LoRA.

Examples
--------
Single-node 8 GPU training with torchrun::

    torchrun --nproc_per_node=8 -m depth_anything_3.training.train_da3_eomt_tslora_stage1 \
      --config src/depth_anything_3/configs/da3_eomt_tslora/base_coco_stage1.yaml \
      --trainer.max_epochs=50 --data.batch_size=4 --logger.project=da3_eomt_tslora \
      --logger.name=da3_base_tsLoRA_coco_stage1

Single GPU debugging::

    python -m depth_anything_3.training.train_da3_eomt_tslora_stage1 \
      --config src/depth_anything_3/configs/da3_eomt_tslora/base_coco_stage1.yaml \
      --trainer.devices=1 --trainer.accelerator=gpu --trainer.max_epochs=2 \
      --data.batch_size=2 --logger.project=da3_eomt_tslora --logger.name=debug_stage1_single_gpu

Resume from a Stage-1 checkpoint::

    python -m depth_anything_3.training.train_da3_eomt_tslora_stage1 \
      --config src/depth_anything_3/configs/da3_eomt_tslora/base_coco_stage1.yaml \
      --trainer.devices=1 --trainer.accelerator=gpu --trainer.max_epochs=50 \
      --data.batch_size=4 --trainer.resume_from_checkpoint=path/to/stage1_ckpt.ckpt \
      --logger.project=da3_eomt_tslora --logger.name=da3_base_tsLoRA_coco_stage1_resume
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Dict
import yaml

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from depth_anything_3.training.da3_eomt_tslora_module import DA3EoMTTSLoRALightning
from depth_anything_3.utils.checkpoint_utils import (
    load_da3_pretrained_backbone,
    resolve_da3_ckpt_path,
)


# ------------------------------
# CLI + config handling
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 COCO panoptic training for DA3+EoMT TS-LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--trainer.max_epochs", dest="trainer_max_epochs", type=int, help="Override max epochs")
    parser.add_argument("--trainer.devices", dest="trainer_devices", type=int, help="Override device count")
    parser.add_argument(
        "--trainer.accelerator", dest="trainer_accelerator", type=str, help="Override accelerator type"
    )
    parser.add_argument("--data.batch_size", dest="data_batch_size", type=int, help="Override training batch size")
    parser.add_argument("--logger.project", dest="logger_project", type=str, help="Override WandB project")
    parser.add_argument("--logger.name", dest="logger_name", type=str, help="Override WandB run name")
    return parser.parse_args()


def _import_from_path(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _import_builder(path: str):
    module_name, func_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    def _set(path: str, value: Any) -> None:
        if value is None:
            return
        keys = path.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    _set("trainer.max_epochs", getattr(args, "trainer_max_epochs"))
    _set("trainer.devices", getattr(args, "trainer_devices"))
    _set("trainer.accelerator", getattr(args, "trainer_accelerator"))
    _set("data.batch_size", getattr(args, "data_batch_size"))
    _set("logger.project", getattr(args, "logger_project"))
    _set("logger.name", getattr(args, "logger_name"))
    return cfg


# ------------------------------
# Builders
# ------------------------------

def build_datamodule(cfg: Dict[str, Any]):
    module_path = cfg.get("module")
    if module_path is None:
        raise ValueError("data.module must be provided")
    module_cls = _import_from_path(module_path)
    return module_cls(
        path=cfg.get("path"),
        stuff_classes=cfg.get("stuff_classes"),
        num_workers=cfg.get("num_workers", 8),
        batch_size=cfg.get("batch_size", 4),
        img_size=tuple(cfg.get("img_size", [640, 640])),
        num_classes=cfg.get("num_classes", 133),
        color_jitter_enabled=cfg.get("color_jitter_enabled", False),
        scale_range=tuple(cfg.get("scale_range", [0.1, 2.0])),
        check_empty_targets=cfg.get("check_empty_targets", True),
    )


def build_lightning_module(cfg: Dict[str, Any]) -> DA3EoMTTSLoRALightning:
    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})

    builder_path = model_cfg.get("builder")
    if builder_path is None:
        raise ValueError("model.builder must point to a callable builder")
    builder = _import_builder(builder_path)

    num_classes = trainer_cfg.get("num_classes", data_cfg.get("num_classes", 133))
    network = builder(model_cfg, num_classes=num_classes)
    ckpt_path = resolve_da3_ckpt_path(model_cfg.get("da3_pretrained_path", "") or "")
    if ckpt_path and hasattr(network, "backbone"):
        load_da3_pretrained_backbone(network.backbone, ckpt_path, strict=False)
    else:
        print("😂😂😂ckpt_path is None or backbone is not found!😂😂😂")

    lightning_kwargs = dict(
        network=network,
        img_size=tuple(data_cfg.get("img_size", model_cfg.get("img_size", [640, 640]))),
        num_classes=num_classes,
        attn_mask_annealing_enabled=model_cfg.get("attn_mask_annealing_enabled", True),
        attn_mask_annealing_start_steps=model_cfg.get("attn_mask_annealing_start_steps"),
        attn_mask_annealing_end_steps=model_cfg.get("attn_mask_annealing_end_steps"),
        lr=trainer_cfg.get("lr", 1e-4),
        llrd=trainer_cfg.get("llrd", 1.0),
        llrd_l2_enabled=trainer_cfg.get("llrd_l2_enabled", False),
        lr_mult=trainer_cfg.get("lr_mult", 1.0),
        weight_decay=trainer_cfg.get("weight_decay", 0.05),
        poly_power=trainer_cfg.get("poly_power", 0.9),
        warmup_steps=tuple(trainer_cfg.get("warmup_steps", [1500, 3000])),
        lambda_2d=model_cfg.get("lambda_2d", 1.0),
        lambda_3d=model_cfg.get("lambda_3d", 0.0),
        stage=model_cfg.get("stage", trainer_cfg.get("stage", "stage1_coco_2d")),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        freeze_depth_ray_head=model_cfg.get("freeze_depth_ray_head", True),
        train_lora_only=model_cfg.get("train_lora_only", True),
    )

    return DA3EoMTTSLoRALightning(**lightning_kwargs)


def build_logger(cfg: Dict[str, Any]) -> WandbLogger:
    return WandbLogger(
        project=cfg.get("project", "da3_eomt_tslora"),
        name=cfg.get("name", "da3_eomt_tslora_coco_stage1"),
        tags=cfg.get("tags"),
        entity=cfg.get("entity"),
        resume=cfg.get("resume", "allow"),
    )


def build_trainer(cfg: Dict[str, Any], logger: WandbLogger) -> L.Trainer:
    return L.Trainer(
        max_epochs=cfg.get("max_epochs", 50),
        precision=cfg.get("precision", "16-mixed"),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        accelerator=cfg.get("accelerator", "gpu"),
        devices=cfg.get("devices"),
        logger=logger,
        deterministic=cfg.get("deterministic", False),
    )


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args)

    datamodule = build_datamodule(cfg.get("data", {}))
    lightning_module = build_lightning_module(cfg)
    logger = build_logger(cfg.get("logger", {}))
    trainer = build_trainer(cfg.get("trainer", {}), logger)

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.get("trainer", {}).get("resume_from_checkpoint"))


if __name__ == "__main__":
    main()
