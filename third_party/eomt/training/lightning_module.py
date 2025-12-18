# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import copy
import math
from typing import Any, Dict, Optional, Tuple, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad
import logging

from .two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule
from safetensors.torch import load_file as load_safetensors

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        llrd_l2_enabled: bool,
        lr_mult: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
        da3_key_mapping=False,
        baseline0: Optional[dict] = None,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled
        self.baseline0_cfg = self._build_baseline0_cfg(baseline0 or {})
        self.baseline0_enabled = bool(self.baseline0_cfg["baseline0_enabled"])

        if self.baseline0_enabled:
            self.automatic_optimization = False  # type: ignore[assignment]
            self._freeze_depth_head()

        if hasattr(self.network, "encoder") and hasattr(self.network.encoder, "lora_report"):
            report = getattr(self.network.encoder, "lora_report", {})
            if report.get("modules", 0) > 0:
                rank_zero_info(
                    f"[LoRA] injected modules={report.get('modules')} layers={report.get('layers')} "
                    f"trainable_lora={report.get('trainable_lora')}"
                )

        self.strict_loading = False
        self._depth_teacher = None
        self._warned_missing_depth = False

        if ckpt_path:
            # 准备键名映射
            key_mapping = None
            if da3_key_mapping:
                key_mapping = lambda k: self._map_da3_key(k)
            
            if delta_weights:
                logging.info("Delta weights mode")
                self._zero_init_outside_encoder(skip_class_head=not load_ckpt_class_head)
                current_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
                if not load_ckpt_class_head:
                    current_state_dict = {
                        k: v
                        for k, v in current_state_dict.items()
                        if "class_head" not in k and "class_predictor" not in k
                    }
                ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head, key_mapping)  # ← 传递 key_mapping
                combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
                incompatible_keys = self.load_state_dict(combined_state_dict, strict=False)
                self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head, allow_missing_for_da3=da3_key_mapping)
            else:
                # 正常加载模式
                ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head, key_mapping)  # ← 传递 key_mapping
                # DA3 特殊处理：移除 pos_embed 中的 cls token
                if da3_key_mapping:
                    ckpt = self._adjust_da3_pos_embed(ckpt)
                incompatible_keys = self.load_state_dict(ckpt, strict=False)
                self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head, allow_missing_for_da3=da3_key_mapping)

        self.log = torch.compiler.disable(self.log)  # type: ignore

    def _map_da3_key(self, key: str) -> str:
        """将 DA3 的键名映射到 EoMT 的键名"""
        # model.backbone.pretrained.* → network.encoder.backbone.*
        if key.startswith("model.backbone.pretrained."):
            return key.replace("model.backbone.pretrained.", "network.encoder.backbone.")
        # model.backbone.* → network.encoder.backbone.*
        elif key.startswith("model.backbone."):
            return key.replace("model.backbone.", "network.encoder.backbone.")
        # 跳过 depth head
        elif key.startswith("model.head."):
            return None  # 不加载
        return key
        

    def _adjust_da3_pos_embed(self, ckpt):
        """调整 DA3 的 pos_embed 以匹配当前模型
        
        DA3 的 pos_embed 包含 cls token: [1, 1+N, C]
        timm 的 pos_embed 只包含 patches: [1, N, C]
        """
        pos_embed_key = "network.encoder.backbone.pos_embed"
        
        if pos_embed_key in ckpt:
            ckpt_pos_embed = ckpt[pos_embed_key]  # DA3: [1, 1370, 768]
            
            try:
                model_pos_embed = self.network.encoder.backbone.pos_embed  # timm: [1, 1369, 768]
                
                if ckpt_pos_embed.shape[1] == model_pos_embed.shape[1] + 1:
                    # DA3 多了一个 cls token，移除它
                    logging.info(f"Removing cls token from DA3 pos_embed: {ckpt_pos_embed.shape} -> {model_pos_embed.shape}")
                    ckpt[pos_embed_key] = ckpt_pos_embed[:, 1:, :]  # 移除第一个 token (cls)
                    logging.info(f"Adjusted pos_embed shape: {ckpt[pos_embed_key].shape}")
                elif ckpt_pos_embed.shape != model_pos_embed.shape:
                    logging.warning(f"pos_embed shape mismatch: checkpoint {ckpt_pos_embed.shape} vs model {model_pos_embed.shape}")
            except Exception as e:
                logging.warning(f"Could not adjust pos_embed: {e}")
        
        return ckpt  

    def _build_baseline0_cfg(self, cfg: dict) -> dict:
        """Assemble baseline-0 configuration with safe defaults."""

        defaults = {
            "baseline0_enabled": False,
            "depth_teacher_ckpt": "",
            "depth_teacher_config": "",
            "lambda_old": 1.0,
            "beta_depth_grad": 0.5,
            "pcgrad_enabled": True,
            "pcgrad_eps": 1e-12,
            "backbone_lr_mult": 0.1,
            "lora_lr_mult": 5.0,
            "freeze_depth_head": True,
            "unfreeze_encoder_layers": 12,
            "warmup_epochs": 1,
            "anchor_on": "depth",
            "log_conflict_stats": True,
        }
        merged = {**defaults, **cfg}
        merged["unfreeze_encoder_layers"] = int(merged["unfreeze_encoder_layers"])
        return merged

    def _freeze_depth_head(self):
        if not self.baseline0_cfg["freeze_depth_head"]:
            return

        encoder = getattr(self.network, "encoder", None)
        depth_head = None
        if encoder is not None:
            if hasattr(encoder, "depth_head"):
                depth_head = encoder.depth_head
            elif hasattr(encoder, "head"):
                depth_head = encoder.head
            elif hasattr(encoder, "da3"):
                depth_head = encoder.da3.head

        if depth_head is None:
            return

        for p in depth_head.parameters():
            p.requires_grad = False

    def _set_encoder_trainable_layers(self, num_layers: int):
        encoder = getattr(self.network, "encoder", None)
        backbone = getattr(encoder, "backbone", None) if encoder is not None else None
        if backbone is None or not hasattr(backbone, "blocks"):
            return

        total_blocks = len(backbone.blocks)
        start_idx = max(total_blocks - num_layers, 0)
        for idx, block in enumerate(backbone.blocks):
            requires_grad = idx >= start_idx
            for p in block.parameters():
                p.requires_grad = requires_grad

        if hasattr(backbone, "patch_embed"):
            for p in backbone.patch_embed.parameters():
                p.requires_grad = num_layers > 0
        if hasattr(backbone, "pos_embed"):
            backbone.pos_embed.requires_grad = num_layers > 0

    def _is_lora_param(self, name: str) -> bool:
        lowered = name.lower()
        return "lora" in lowered or "adapter" in lowered

    def _build_optimizer_and_scheduler(
        self, ignore_prefixes: Optional[tuple[str, ...]] = None
    ) -> tuple[AdamW, TwoStageWarmupPolySchedule, int]:
        from third_party.eomt.models.da3_adapter import DA3BackboneAdapter, LoRALinear

        encoder = getattr(self.network, "encoder", None)
        backbone = getattr(encoder, "backbone", None) if encoder is not None else None
        adapter = encoder if isinstance(encoder, DA3BackboneAdapter) else None

        encoder_param_ids = (
            {id(p) for _, p in encoder.named_parameters()} if encoder is not None else set()
        )
        backbone_param_ids = (
            {id(p) for _, p in backbone.named_parameters()}
            if backbone is not None and hasattr(backbone, "named_parameters")
            else set()
        )
        backbone_blocks = len(backbone.blocks) if backbone is not None and hasattr(backbone, "blocks") else 0
        l2_blocks = (
            torch.arange(backbone_blocks - getattr(self.network, "num_blocks", 0), backbone_blocks).tolist()
            if backbone_blocks > 0 and hasattr(self.network, "num_blocks")
            else []
        )

        backbone_lr_mult = self.baseline0_cfg["backbone_lr_mult"] if self.baseline0_enabled else 1.0
        lora_lr_mult = self.baseline0_cfg["lora_lr_mult"] if self.baseline0_enabled else 1.0

        param_groups: list[dict[str, Any]] = []
        backbone_group_count = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ignore_prefixes is not None and any(name.startswith(pfx) for pfx in ignore_prefixes):
                continue
            if adapter is not None and adapter.is_depth_head_param(param):
                continue

            is_backbone = id(param) in backbone_param_ids
            is_encoder = id(param) in encoder_param_ids
            block_idx = adapter.block_index_for_param(param) if adapter is not None else None
            is_lora = self._is_lora_param(name)

            lr = self.lr
            weight_decay = self.weight_decay

            if adapter is not None:
                if block_idx is not None:
                    llrd_factor = self.llrd ** (backbone_blocks - 1 - block_idx) if backbone_blocks > 0 else 1.0
                    lr = self.lr * (lora_lr_mult if is_lora else backbone_lr_mult) * llrd_factor
                    if is_lora:
                        weight_decay = 0.0
                    backbone_group_count += 1
                else:
                    lr = self.lr
            else:
                if is_backbone:
                    block_idx = None
                    if ".blocks." in name:
                        try:
                            after = name.split(".blocks.", 1)[1]
                            block_idx = int(after.split(".", 1)[0])
                        except Exception:
                            block_idx = None
                    llrd_factor = (
                        self.llrd ** (backbone_blocks - 1 - block_idx)
                        if block_idx is not None and backbone_blocks > 0
                        else 1.0
                    )
                    if block_idx in l2_blocks and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0)):
                        llrd_factor = 1.0
                    base_mult = lora_lr_mult if is_lora else backbone_lr_mult
                    lr = self.lr * base_mult * llrd_factor
                    if block_idx is None and self.lr_mult != 1.0:
                        lr *= self.lr_mult
                    if is_lora:
                        weight_decay = 0.0
                    backbone_group_count += 1
                elif is_encoder:
                    lr = self.lr * backbone_lr_mult
                    backbone_group_count += 1

            group = {"params": [param], "lr": lr, "name": name, "weight_decay": weight_decay}
            param_groups.append(group)

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=backbone_group_count,
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )
        return optimizer, scheduler, backbone_group_count

    def _maybe_forward_depth(self, imgs: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        """Try to obtain a student depth prediction without changing existing flows."""

        encoder = getattr(self.network, "encoder", None)
        if encoder is None:
            return None

        # Try explicit depth forward
        for candidate in ("forward_depth", "forward_with_depth"):
            if hasattr(encoder, candidate):
                try:
                    depth_out = getattr(encoder, candidate)(imgs, **kwargs)
                    depth_pred, _ = self._extract_depth_from_output(depth_out)
                    if depth_pred is not None:
                        return depth_pred
                except Exception:
                    pass

        # Try regular forward with depth hints
        try:
            depth_out = encoder(imgs, return_depth=True, **kwargs)
            depth_pred, _ = self._extract_depth_from_output(depth_out)
            if depth_pred is not None:
                return depth_pred
        except Exception:
            pass

        return None

    def _extract_depth_from_output(
        self, output: Any
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract depth tensor and optional validity mask from arbitrary outputs."""

        depth = None
        valid = None
        if output is None:
            return None, None

        if isinstance(output, dict):
            for key in ("depth", "depths", "pred_depth", "pred_depths"):
                if key in output:
                    depth = output[key]
                    break
            for key in ("valid_mask", "depth_mask", "mask"):
                if key in output:
                    valid = output[key]
                    break
        else:
            if hasattr(output, "depth"):
                depth = output.depth
            if hasattr(output, "valid_mask"):
                valid = output.valid_mask

        return depth, valid

    def _split_panoptic_outputs(
        self, outputs: Any
    ) -> tuple[tuple[list[torch.Tensor], list[torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Handle flexible outputs that may bundle panoptic logits with depth."""

        depth_pred, valid_mask = None, None
        panoptic = outputs
        if isinstance(outputs, dict):
            panoptic = outputs.get("panoptic", panoptic)
            depth_pred, valid_mask = self._extract_depth_from_output(outputs)

        if not isinstance(panoptic, (list, tuple)) or len(panoptic) != 2:
            raise ValueError("Panoptic outputs must be a tuple of (mask_logits, class_logits)")

        return cast(tuple[list[torch.Tensor], list[torch.Tensor]], panoptic), depth_pred, valid_mask

    def depth_teacher_forward(self, batch) -> Dict[str, torch.Tensor]:
        """Run the frozen depth teacher with lazy initialization."""

        if not self.baseline0_cfg["depth_teacher_ckpt"]:
            raise ValueError("depth_teacher_ckpt must be set when baseline0 is enabled")

        if self._depth_teacher is None:
            from third_party.eomt.models.da3_adapter import DA3BackboneAdapter

            if isinstance(getattr(self.network, "encoder", None), DA3BackboneAdapter):
                teacher = self.network.encoder.build_teacher_copy(self.baseline0_cfg["depth_teacher_ckpt"])
            else:
                cfg_path = self.baseline0_cfg.get("depth_teacher_config") or ""
                teacher = DA3BackboneAdapter(
                    da3_config_path=cfg_path,
                    da3_ckpt_path=self.baseline0_cfg["depth_teacher_ckpt"],
                    lora={"enabled": False},
                    freeze_depth_head=True,
                )

            teacher.eval()
            teacher.requires_grad_(False)
            self._depth_teacher = teacher.to(self.device)

        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        if imgs.ndim == 4:
            imgs_teacher = imgs[:, None, ...]
        else:
            imgs_teacher = imgs

        imgs_teacher = imgs_teacher.to(self.device)

        with torch.no_grad():
            depth_t = self._depth_teacher.forward_depth(imgs_teacher)
        return {"depth_t": depth_t, "valid_mask": None}

    def depth_anchor_loss(
        self, depth_s: torch.Tensor, depth_t: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute scale-and-shift invariant depth anchor with optional gradient term."""

        if depth_s.dim() == 4:
            depth_s = depth_s.squeeze(1)
        if depth_t.dim() == 4:
            depth_t = depth_t.squeeze(1)
        if valid_mask is not None and valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)

        if valid_mask is None:
            valid_mask = torch.isfinite(depth_s) & torch.isfinite(depth_t)
        eps = 1e-6

        losses_ssi, losses_grad = [], []
        for b in range(depth_s.shape[0]):
            mask = valid_mask[b]
            if mask.sum() == 0:
                losses_ssi.append(torch.tensor(0.0, device=depth_s.device))
                losses_grad.append(torch.tensor(0.0, device=depth_s.device))
                continue
            ds = depth_s[b][mask]
            dt = depth_t[b][mask]
            n = ds.numel()
            sum_ds = ds.sum()
            sum_dt = dt.sum()
            sum_ds2 = (ds * ds).sum()
            sum_dsdt = (ds * dt).sum()
            denom = n * sum_ds2 - sum_ds**2
            if torch.abs(denom) < eps:
                a = torch.tensor(1.0, device=depth_s.device, dtype=depth_s.dtype)
                b_val = torch.tensor(0.0, device=depth_s.device, dtype=depth_s.dtype)
            else:
                a = (n * sum_dsdt - sum_ds * sum_dt) / (denom + eps)
                b_val = (sum_dt - a * sum_ds) / n

            aligned = a * depth_s[b] + b_val
            ssi_l1 = torch.mean(torch.abs(aligned[mask] - depth_t[b][mask]))

            grad_s = torch.stack(
                [
                    aligned[:, 1:] - aligned[:, :-1],
                    aligned[1:, :] - aligned[:-1, :],
                ],
                dim=0,
            )
            grad_t = torch.stack(
                [
                    depth_t[b][:, 1:] - depth_t[b][:, :-1],
                    depth_t[b][1:, :] - depth_t[b][:-1, :],
                ],
                dim=0,
            )
            if valid_mask is not None:
                grad_mask = torch.stack(
                    [
                        mask[:, 1:] & mask[:, :-1],
                        mask[1:, :] & mask[:-1, :],
                    ],
                    dim=0,
                )
                grad_diff = torch.abs(grad_s - grad_t)[grad_mask]
            else:
                grad_diff = torch.abs(grad_s - grad_t)
            grad_l1 = grad_diff.mean() if grad_diff.numel() > 0 else torch.tensor(0.0, device=depth_s.device)

            losses_ssi.append(ssi_l1)
            losses_grad.append(grad_l1)

        depth_ssi_l1 = torch.stack(losses_ssi).mean()
        depth_grad_l1 = torch.stack(losses_grad).mean()
        loss_old = self.baseline0_cfg["lambda_old"] * (
            depth_ssi_l1 + self.baseline0_cfg["beta_depth_grad"] * depth_grad_l1
        )
        stats = {
            "depth_ssi_l1": depth_ssi_l1.detach(),
            "depth_grad_l1": depth_grad_l1.detach(),
            "depth_old_loss": loss_old.detach(),
        }
        return loss_old, stats

    def pcgrad_step_on_encoder(
        self, loss_pan: torch.Tensor, loss_old: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """Perform one-sided PCGrad on encoder parameters before stepping."""

        encoder = getattr(self.network, "encoder", None)
        if encoder is None:
            raise RuntimeError("Encoder is required for PCGrad updates")

        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)

        # Depth branch
        self.manual_backward(loss_old, retain_graph=True)
        g_old = [p.grad.clone().float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32) for p in encoder_params]
        for p in encoder_params:
            if p.grad is not None:
                p.grad = None

        # Panoptic branch (also updates heads)
        self.manual_backward(loss_pan)
        g_pan = [p.grad.clone().float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32) for p in encoder_params]

        dot = torch.stack([torch.dot(gp.flatten(), go.flatten()) for gp, go in zip(g_pan, g_old)]).sum()
        norm_old_sq = torch.stack([go.flatten().dot(go.flatten()) for go in g_old]).sum()

        conflict_flag = bool((dot < 0).item())
        if conflict_flag:
            coef = -dot / (norm_old_sq + self.baseline0_cfg["pcgrad_eps"])
            for idx, p in enumerate(encoder_params):
                projected = g_pan[idx] + coef * g_old[idx]
                p.grad = projected.to(p.dtype)
        else:
            for idx, p in enumerate(encoder_params):
                p.grad = (g_pan[idx] + g_old[idx]).to(p.dtype)

        if self.baseline0_cfg["log_conflict_stats"]:
            self.log(
                "pcgrad_conflict",
                torch.tensor(float(conflict_flag), device=self.device),
                on_step=True,
                prog_bar=False,
            )
            self.log("pcgrad_dot", dot, on_step=True, prog_bar=False)
            self.log(
                "pcgrad_coef",
                (-dot / (norm_old_sq + self.baseline0_cfg["pcgrad_eps"]))
                if conflict_flag
                else torch.tensor(0.0, device=self.device),
                on_step=True,
                prog_bar=False,
            )
            self.log("pcgrad_norm_old", norm_old_sq.sqrt(), on_step=True, prog_bar=False)
            self.log(
                "pcgrad_norm_pan",
                torch.stack([gp.flatten().dot(gp.flatten()) for gp in g_pan]).sum().sqrt(),
                on_step=True,
                prog_bar=False,
            )

        optimizer.step()

    def configure_optimizers(self):
        optimizer, scheduler, num_backbone = self._build_optimizer_and_scheduler()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
            "monitor": None,
        }

    def forward(self, imgs, return_depth: bool = False, **kwargs):
        x = imgs / 255.0
        outputs = self.network(x, **kwargs)

        if return_depth:
            depth = None
            if isinstance(outputs, dict):
                depth, _ = self._extract_depth_from_output(outputs)
            if depth is None:
                depth = self._maybe_forward_depth(x, **kwargs)
            if depth is not None:
                return {"panoptic": outputs, "depth": depth}

        return outputs

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        if not self.baseline0_enabled:
            mask_logits_per_block, class_logits_per_block = self(imgs)

            losses_all_blocks = {}
            for i, (mask_logits, class_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block))
            ):
                losses = self.criterion(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=targets,
                )
                block_postfix = self.block_postfix(i)
                losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
                losses_all_blocks |= losses

            return self.criterion.loss_total(losses_all_blocks, self.log)

        # Baseline-0 path (manual optimization)
        optim = self.optimizers()
        schedulers = self.lr_schedulers()
        self._freeze_depth_head()

        panoptic_outputs = self(imgs)
        panoptic_outputs, _, _ = self._split_panoptic_outputs(panoptic_outputs)
        depth_s = None
        valid_s = None
        if hasattr(self.network, "encoder") and hasattr(self.network.encoder, "forward_depth"):
            depth_s = self.network.encoder.forward_depth(imgs)
        mask_logits_per_block, class_logits_per_block = panoptic_outputs

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses
        loss_pan = self.criterion.loss_total(losses_all_blocks, self.log)

        # Teacher depth
        loss_old = torch.tensor(0.0, device=self.device)
        depth_stats: Dict[str, torch.Tensor] = {}
        if self.baseline0_cfg["anchor_on"] == "depth":
            teacher_out = self.depth_teacher_forward(batch)
            depth_t = teacher_out["depth_t"]
            valid_t = teacher_out.get("valid_mask", None)

            if depth_s is None:
                if not self._warned_missing_depth:
                    logging.warning("Baseline-0 enabled but student depth head is missing; skipping anchor loss.")
                    self._warned_missing_depth = True
            else:
                if depth_s.ndim == 5:
                    depth_s = depth_s.view(-1, *depth_s.shape[2:])
                if valid_s is None:
                    valid_s = valid_t
                loss_old, depth_stats = self.depth_anchor_loss(depth_s, depth_t, valid_s if valid_s is not None else valid_t)
                for key, val in depth_stats.items():
                    self.log(f"baseline0/{key}", val, on_step=True, on_epoch=True, prog_bar=False)

        total_loss = loss_pan + loss_old
        warmup = self.baseline0_cfg["warmup_epochs"]

        if self.current_epoch < warmup:
            self._set_encoder_trainable_layers(0)
            optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_pan)
            optim.step()
        else:
            self._set_encoder_trainable_layers(self.baseline0_cfg["unfreeze_encoder_layers"])
            if self.baseline0_cfg["pcgrad_enabled"]:
                self.pcgrad_step_on_encoder(loss_pan, loss_old, optim)
            else:
                optim.zero_grad(set_to_none=True)
                self.manual_backward(total_loss)
                optim.step()

        if schedulers is not None:
            if isinstance(schedulers, list):
                for sch in schedulers:
                    sch.step()
            else:
                schedulers.step()

        self.log(
            "train_loss_panoptic",
            loss_pan.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if loss_old is not None:
            self.log(
                "train_loss_depth_anchor",
                loss_old.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log(
            "train_loss_total",
            total_loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss.detach()

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )

    def init_metrics_panoptic(self, thing_classes, stuff_classes, num_blocks):
        self.metrics = nn.ModuleList(
            [
                PanopticQuality(
                    thing_classes,
                    stuff_classes + [self.num_classes],
                    return_sq_and_rq=True,
                    return_per_class=True,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    @torch.compiler.disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx,
    ):
        self.metrics[block_idx].update(preds, targets)

    @torch.compiler.disable
    def update_metrics_panoptic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        is_crowds: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            metric = self.metrics[block_idx]
            flatten_pred = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                preds[i][None, ...],
                metric.void_color,
                metric.allow_unknown_preds_category,
            )[0]
            flatten_target = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                targets[i][None, ...],
                metric.void_color,
                True,
            )[0]

            pred_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_pred)
            )
            target_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_target)
            )
            intersection_matrix = torch.transpose(
                torch.stack((flatten_pred, flatten_target), -1), -1, -2
            )
            intersection_areas = cast(
                dict[tuple[_Color, _Color], torch.Tensor],
                _get_color_areas(intersection_matrix),
            )

            pred_segment_matched = set()
            target_segment_matched = set()
            for pred_color, target_color in intersection_areas:
                if is_crowds[i][target_color[1]]:
                    continue
                if target_color == metric.void_color:
                    continue
                if pred_color[0] != target_color[0]:
                    continue
                iou = _calculate_iou(
                    pred_color,
                    target_color,
                    pred_areas,
                    target_areas,
                    intersection_areas,
                    metric.void_color,
                )
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    metric.iou_sum[continuous_id] += iou
                    metric.true_positives[continuous_id] += 1

            false_negative_colors = set(target_areas) - target_segment_matched
            false_positive_colors = set(pred_areas) - pred_segment_matched

            false_negative_colors.discard(metric.void_color)
            false_positive_colors.discard(metric.void_color)

            for target_color in list(false_negative_colors):
                void_target_area = intersection_areas.get(
                    (metric.void_color, target_color), 0
                )
                if void_target_area / target_areas[target_color] > 0.5:
                    false_negative_colors.discard(target_color)

            crowd_by_cat_id = {}
            for false_negative_color in false_negative_colors:
                if is_crowds[i][false_negative_color[1]]:
                    crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                    continue

                continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
                metric.false_negatives[continuous_id] += 1

            for pred_color in list(false_positive_colors):
                pred_void_crowd_area = intersection_areas.get(
                    (pred_color, metric.void_color), 0
                )

                if pred_color[0] in crowd_by_cat_id:
                    crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                    pred_void_crowd_area += intersection_areas.get(
                        (pred_color, crowd_color), 0
                    )

                if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                    false_positive_colors.discard(pred_color)

            for false_positive_color in false_positive_colors:
                continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
                metric.false_positives[continuous_id] += 1

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

    def _on_eval_epoch_end_instance(self, log_prefix):
        for i, metric in enumerate(self.metrics):  # type: ignore
            results = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(
                f"metrics/{log_prefix}_ap_all{block_postfix}",
                results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                results["map_75"],
            )

    def _on_eval_epoch_end_panoptic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            tp = metric.true_positives.to(torch.float32)
            fp = metric.false_positives.to(torch.float32)
            fn = metric.false_negatives.to(torch.float32)
            iou_sum = metric.iou_sum.to(torch.float32)

            den = tp + 0.5 * fp + 0.5 * fn

            pq = torch.zeros_like(den)
            sq = torch.zeros_like(den)
            rq = torch.zeros_like(den)

            non_zero_den = den > 0
            pq[non_zero_den] = iou_sum[non_zero_den] / den[non_zero_den]
            rq[non_zero_den] = tp[non_zero_den] / den[non_zero_den]

            tp_mask = tp > 0
            sq[tp_mask] = iou_sum[tp_mask] / tp[tp_mask]

            result = torch.stack((pq, sq, rq), dim=-1)[:-1]
            metric.reset()

            pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx in range(len(pq)):
                    self.log(
                        f"metrics/{log_prefix}_pq_class_{class_idx}{block_postfix}",
                        pq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_sq_class_{class_idx}{block_postfix}",
                        sq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_rq_class_{class_idx}{block_postfix}",
                        rq[class_idx],
                    )

            self.log(
                f"metrics/{log_prefix}_pq_all{block_postfix}",
                pq.mean(),
            )
            self.log(f"metrics/{log_prefix}_sq_all{block_postfix}", sq.mean())
            self.log(f"metrics/{log_prefix}_rq_all{block_postfix}", rq.mean())

            num_things = len(metric.things)
            pq_things, sq_things, rq_things = (
                result[:num_things, 0],
                result[:num_things, 1],
                result[:num_things, 2],
            )
            pq_stuff, sq_stuff, rq_stuff = (
                result[num_things:, 0],
                result[num_things:, 1],
                result[num_things:, 2],
            )

            self.log(
                f"metrics/{log_prefix}_pq_things{block_postfix}",
                pq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_things{block_postfix}",
                sq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_things{block_postfix}",
                rq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_pq_stuff{block_postfix}",
                pq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_stuff{block_postfix}",
                sq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_stuff{block_postfix}",
                rq_stuff.mean(),
            )

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_instance(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
                f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
                f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
                f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_panoptic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}PQ All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_all'] * 100:.1f} | "
                f"PQ Things: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_things'] * 100:.1f} | "
                f"PQ Stuff: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_stuff'] * 100:.1f}{reset}"
            )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self, mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh
    ):
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.long,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.long,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return preds_list

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_panoptic(targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = -torch.ones(
                (*target["masks"].shape[-2:], 2),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[:, :, 0] = torch.where(
                    mask, target["labels"][i], per_pixel_target[:, :, 0]
                )

                per_pixel_target[:, :, 1] = torch.where(
                    mask,
                    torch.tensor(i, device=target["masks"].device),
                    per_pixel_target[:, :, 1],
                )

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _zero_init_outside_encoder(
        self, encoder_prefix="network.encoder.", skip_class_head=False
    ):
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, p in self.named_parameters():
                total += p.numel()
                if not name.startswith(encoder_prefix):
                    if skip_class_head and (
                        "class_head" in name or "class_predictor" in name
                    ):
                        continue
                    p.zero_()
                    zeroed += p.numel()
            msg = f"Zeroed {zeroed:,} / {total:,} parameters (everything not under '{encoder_prefix}'"
            if skip_class_head:
                msg += ", skipping class head"
            msg += ")"
            logging.info(msg)

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for k in state_dict1.keys():
            if k not in state_dict2:
                raise KeyError(f"Key {k} not found in second state_dict")

            if state_dict1[k].shape != state_dict2[k].shape:
                raise ValueError(
                    f"Shape mismatch at {k}: "
                    f"{state_dict1[k].shape} vs {state_dict2[k].shape}"
                )

            summed[k] = state_dict1[k] + state_dict2[k]

        return summed

    # def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
    #     ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    #     if "state_dict" in ckpt:
    #         ckpt = ckpt["state_dict"]
    #     ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
    #     if not load_ckpt_class_head:
    #         ckpt = {
    #             k: v
    #             for k, v in ckpt.items()
    #             if "class_head" not in k and "class_predictor" not in k
    #         }
    #     logging.info(f"Loaded {len(ckpt)} keys")
    #     return ckpt
    def _load_ckpt(self, ckpt_path, load_ckpt_class_head, key_mapping=None):
        """
        Load checkpoint from either PyTorch (.pth, .ckpt) or SafeTensors (.safetensors) format.
        
        Args:
            ckpt_path: Path to checkpoint file
            load_ckpt_class_head: Whether to load class head weights
            key_mapping: Optional callable to map checkpoint keys (returns None to skip)
            
        Returns:
            State dict
        """
        # 检测文件格式
        import os
        file_ext = os.path.splitext(ckpt_path)[1].lower()
        
        if file_ext == '.safetensors':
            # 加载 SafeTensors 格式
            logging.info(f"Loading SafeTensors checkpoint from {ckpt_path}")
            ckpt = load_safetensors(ckpt_path)
        else:
            # 加载 PyTorch 格式 (.pth, .ckpt, etc.)
            logging.info(f"Loading PyTorch checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            
            # PyTorch checkpoints 可能包含 'state_dict' 键
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
        
        # 移除不需要的键
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        
        # 可选：不加载 class head
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        
        logging.info(f"Loaded {len(ckpt)} keys from {file_ext} format")
        
        # 应用键名映射（如果提供）
        if key_mapping is not None:
            logging.info("Applying key mapping...")
            mapped_ckpt = {}
            skipped_keys = []
            for k, v in ckpt.items():
                new_key = key_mapping(k)
                if new_key is None:
                    skipped_keys.append(k)
                    continue  # 跳过这个键
                mapped_ckpt[new_key] = v
            
            logging.info(f"Mapped {len(mapped_ckpt)} keys, skipped {len(skipped_keys)} keys")
            if skipped_keys:
                logging.info(f"Skipped keys: {skipped_keys[:10]}...")  # 只显示前10个
            ckpt = mapped_ckpt
        
        return ckpt

    # def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
    #     if incompatible_keys.missing_keys:
    #         if not load_ckpt_class_head:
    #             missing_keys = [
    #                 key
    #                 for key in incompatible_keys.missing_keys
    #                 if "class_head" not in key and "class_predictor" not in key
    #             ]
    #         else:
    #             missing_keys = incompatible_keys.missing_keys
    #         if missing_keys:
    #             raise ValueError(f"Missing keys: {missing_keys}")
    #     if incompatible_keys.unexpected_keys:
    #         raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head, allow_missing_for_da3=False):
        """
        检查不兼容的 keys
        
        Args:
            incompatible_keys: PyTorch load_state_dict 返回的不兼容 keys
            load_ckpt_class_head: 是否加载 class head
            allow_missing_for_da3: 是否允许 DA3 迁移学习的 missing keys
        """
        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "class_head" not in key and "class_predictor" not in key
                ]
            else:
                missing_keys = incompatible_keys.missing_keys
            
            # 如果是 DA3 迁移学习，过滤掉预期的 missing keys
            if allow_missing_for_da3:
                expected_missing_da3 = [
                    'network.attn_mask_probs',           # EoMT 特有
                    'network.encoder.pixel_mean',        # 预处理参数
                    'network.encoder.pixel_std',         # 预处理参数
                    'network.encoder.backbone.reg_token', # Register token (DA3没有)
                    'network.q.weight',                  # Query embeddings
                    'network.class_head',                # Class head
                ]
                
                # 同时允许 mask_head 和 upscale 的所有子参数
                filtered_missing = []
                for key in missing_keys:
                    is_expected = False
                    # 检查是否是预期的 missing key
                    for expected in expected_missing_da3:
                        if key == expected or key.startswith(expected):
                            is_expected = True
                            break
                    # 检查是否是 mask_head 或 upscale 的参数
                    if 'mask_head' in key or 'upscale' in key:
                        is_expected = True
                    
                    if not is_expected:
                        filtered_missing.append(key)
                
                missing_keys = filtered_missing
                
                if missing_keys:
                    logging.warning(f"Unexpected missing keys (not from DA3): {missing_keys}")
                else:
                    logging.info(f"All missing keys are expected for DA3 transfer learning")
            
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
        
        if incompatible_keys.unexpected_keys:
            # Unexpected keys 通常可以忽略（来自 checkpoint 但模型不需要的参数）
            logging.warning(f"Unexpected keys (will be ignored): {incompatible_keys.unexpected_keys[:10]}...")
            # 如果不是 DA3 迁移，才抛出错误
            if not allow_missing_for_da3:
                raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
