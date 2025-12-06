from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn
from torch.optim import AdamW

from third_party.eomt.training.mask_classification_loss import MaskClassificationLoss
from third_party.eomt.training.two_stage_warmup_poly_schedule import (
    TwoStageWarmupPolySchedule,
)

from depth_anything_3.model.segmentation.head_eomt_adapter import EoMTSegHead


class DA3SegPanopticModule(pl.LightningModule):
    """Stage-1 LightningModule for DA3 + segmentation branch panoptic training."""

    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool = True,
        num_masked_layers: int | None = None,
        mask_annealing_poly_factor: float = 0.9,
        lr: float = 2e-4,
        llrd: float = 0.8,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        warmup_steps: tuple[int, int] = (500, 1000),
        poly_power: float = 0.9,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        num_queries: int = 100,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.mask_annealing_poly_factor = mask_annealing_poly_factor
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.poly_power = poly_power

        self.save_hyperparameters(ignore=["network"])

        self.num_masked_layers = num_masked_layers
        if self.num_masked_layers is None:
            self.num_masked_layers = getattr(self.network.backbone, "num_seg_masked_layers", 0)

        embed_dim = getattr(self.network.backbone, "embed_dim", None)
        patch_grid = None
        if hasattr(self.network.backbone, "patch_embed"):
            patch_grid = getattr(self.network.backbone.patch_embed, "grid_size", None)
        self.seg_head = EoMTSegHead(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            patch_grid=patch_grid or (img_size[0] // 16, img_size[1] // 16),
        )

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=0.1,
        )

        self.attn_mask_annealing_start_steps: List[int] = []
        self.attn_mask_annealing_end_steps: List[int] = []

    def _compute_annealing_schedule(self, total_steps: int) -> None:
        if not self.attn_mask_annealing_enabled or self.num_masked_layers == 0:
            self.attn_mask_annealing_start_steps = []
            self.attn_mask_annealing_end_steps = []
            return

        step_stride = total_steps / (self.num_masked_layers + 1)
        starts, ends = [], []
        for i in range(self.num_masked_layers):
            start = round((i + 1) * step_stride)
            end = round((i + 2) * step_stride)
            starts.append(max(1, start))
            ends.append(min(total_steps, max(start + 1, end)))
        self.attn_mask_annealing_start_steps = starts
        self.attn_mask_annealing_end_steps = ends

    def on_train_start(self) -> None:
        total_steps = int(self.trainer.estimated_stepping_batches)
        self._compute_annealing_schedule(total_steps)

    def _mask_prob_for_layer(self, layer_idx: int, global_step: int) -> float:
        if not self.attn_mask_annealing_enabled or layer_idx >= self.num_masked_layers:
            return 0.0

        start = self.attn_mask_annealing_start_steps[layer_idx]
        end = self.attn_mask_annealing_end_steps[layer_idx]
        if global_step < start:
            return 1.0
        if global_step >= end:
            return 0.0
        t = (global_step - start) / max(1, end - start)
        return float((1.0 - t) ** self.mask_annealing_poly_factor)

    def _build_seg_mask_probs(self, global_step: int) -> List[float]:
        if not self.attn_mask_annealing_enabled or self.num_masked_layers == 0:
            return []
        seg_depth = 0
        if hasattr(self.network, "backbone") and getattr(self.network.backbone, "seg_blocks", None) is not None:
            seg_depth = len(self.network.backbone.seg_blocks) - self.network.backbone.seg_layer_start
        probs = [0.0 for _ in range(max(seg_depth, self.num_masked_layers))]
        for i in range(self.num_masked_layers):
            target_idx = len(probs) - self.num_masked_layers + i
            probs[target_idx] = self._mask_prob_for_layer(i, global_step)
        return probs

    def _extract_seg_tokens(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, dict):
            return output.get("seg_tokens", {})
        return getattr(output, "seg_tokens", {})

    def forward(
        self,
        imgs: torch.Tensor,
        seg_mask_probs: Optional[List[float]] = None,
        seg_head_fn=None,
        apply_seg_head_to_intermediate: bool = True,
        apply_seg_head_to_last: bool = True,
    ):
        return self.network(
            imgs,
            seg_mask_probs=seg_mask_probs,
            seg_head_fn=seg_head_fn,
            apply_seg_head_to_intermediate=apply_seg_head_to_intermediate,
            apply_seg_head_to_last=apply_seg_head_to_last,
        )

    def _collect_head_outputs(
        self, seg_tokens: Dict[str, Any], compute_if_missing: bool
    ) -> list[Dict[str, torch.Tensor]]:
        head_outputs: list[Dict[str, torch.Tensor]] = []
        layers = seg_tokens.get("layers", [])
        for layer_tokens in layers:
            preds = layer_tokens.get("head_outputs")
            if preds is None and compute_if_missing and "G_seg" in layer_tokens and "S" in layer_tokens:
                preds = self.seg_head(layer_tokens["G_seg"], layer_tokens["S"])
            if preds is not None:
                head_outputs.append(preds)

        if not head_outputs and compute_if_missing and "G_seg" in seg_tokens and "S" in seg_tokens:
            head_outputs.append(self.seg_head(seg_tokens["G_seg"], seg_tokens["S"]))

        return head_outputs

    def training_step(self, batch: Any, batch_idx: int):
        imgs, targets = batch
        seg_mask_probs = self._build_seg_mask_probs(self.global_step)
        network_out = self(
            imgs,
            seg_mask_probs=seg_mask_probs,
            seg_head_fn=self.seg_head,
            apply_seg_head_to_intermediate=True,
            apply_seg_head_to_last=True,
        )
        seg_tokens = self._extract_seg_tokens(network_out)
        head_outputs = self._collect_head_outputs(seg_tokens, compute_if_missing=False)
        if not head_outputs:
            head_outputs = self._collect_head_outputs(seg_tokens, compute_if_missing=True)

        losses_all_blocks: Dict[str, torch.Tensor] = {}
        for i, preds in enumerate(head_outputs):
            losses = self.criterion(
                masks_queries_logits=preds["pred_masks"],
                class_queries_logits=preds["pred_logits"],
                targets=targets,
            )
            for key, value in losses.items():
                losses_all_blocks[f"{key}_b{i}"] = value

        total_loss = self.criterion.loss_total(losses_all_blocks, self.log)
        self.log("train_loss", total_loss, prog_bar=True)
        for i, prob in enumerate(seg_mask_probs[-self.num_masked_layers :] if seg_mask_probs else []):
            self.log(f"anneal/p_mask_layer_{i}", prob, prog_bar=False)
        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        imgs, targets = batch
        network_out = self(
            imgs,
            seg_mask_probs=[],
            seg_head_fn=self.seg_head,
            apply_seg_head_to_intermediate=False,
            apply_seg_head_to_last=True,
        )
        seg_tokens = self._extract_seg_tokens(network_out)
        head_outputs = self._collect_head_outputs(seg_tokens, compute_if_missing=True)
        if not head_outputs:
            return torch.tensor(0.0, device=self.device)

        losses_all_blocks: Dict[str, torch.Tensor] = {}
        for i, preds in enumerate(head_outputs):
            losses = self.criterion(
                masks_queries_logits=preds["pred_masks"],
                class_queries_logits=preds["pred_logits"],
                targets=targets,
            )
            for key, value in losses.items():
                losses_all_blocks[f"{key}_b{i}"] = value
        total_loss = self.criterion.loss_total(losses_all_blocks, self.log)
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        param_groups: List[Dict[str, Any]] = []
        vit_depth = len(getattr(self.network.backbone, "blocks", [])) if hasattr(self.network, "backbone") else 0
        num_backbone_params = 0

        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue

            lr = self.lr * self.lr_mult
            decay = self.weight_decay

            if "backbone" in name:
                num_backbone_params += 1
            if "backbone.blocks" in name and vit_depth > 0:
                try:
                    block_idx = int(name.split("backbone.blocks.")[1].split(".")[0])
                    decay_factor = self.llrd ** (vit_depth - block_idx - 1)
                    lr = lr * decay_factor
                except ValueError:
                    pass

            param_groups.append({"params": [param], "lr": lr, "weight_decay": decay})

        optimizer = AdamW(param_groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = TwoStageWarmupPolySchedule(
            optimizer=optimizer,
            num_backbone_params=num_backbone_params,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

