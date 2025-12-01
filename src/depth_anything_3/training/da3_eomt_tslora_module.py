"""Lightning glue for DA3 + EoMT TS-LoRA.

This module wraps the existing EoMT Lightning module so it can host a DA3
backbone with segmentation-aware token-selective LoRA. The intent is to keep
loss computation, annealing, and logging behaviour identical to EoMT while
allowing the network instantiation to differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import lightning
import torch
from lightning.pytorch.loggers import WandbLogger

from third_party.eomt.training.lightning_module import LightningModule as EoMTLightning
from third_party.eomt.training.mask_classification_instance import (
    MaskClassificationInstance,
)

from depth_anything_3.data.scannet_stage2 import MultiViewBatch


@dataclass
class DA3EoMTTSLoRAConfig:
    """Minimal configuration holder for DA3+EoMT Lightning integration."""

    network: lightning.LightningModule
    img_size: tuple[int, int]
    num_classes: int
    attn_mask_annealing_enabled: bool
    attn_mask_annealing_start_steps: list[int] | None
    attn_mask_annealing_end_steps: list[int] | None
    lr: float
    llrd: float
    llrd_l2_enabled: bool
    lr_mult: float
    weight_decay: float
    poly_power: float
    warmup_steps: tuple[int, int]
    ckpt_path: str | None = None
    delta_weights: bool = False
    load_ckpt_class_head: bool = True


def build_da3_eomt_lightning(cfg: DA3EoMTTSLoRAConfig) -> EoMTLightning:
    """Instantiate an EoMT LightningModule with a DA3+TS-LoRA network."""

    return EoMTLightning(
        network=cfg.network,
        img_size=cfg.img_size,
        num_classes=cfg.num_classes,
        attn_mask_annealing_enabled=cfg.attn_mask_annealing_enabled,
        attn_mask_annealing_start_steps=cfg.attn_mask_annealing_start_steps,
        attn_mask_annealing_end_steps=cfg.attn_mask_annealing_end_steps,
        lr=cfg.lr,
        llrd=cfg.llrd,
        llrd_l2_enabled=cfg.llrd_l2_enabled,
        lr_mult=cfg.lr_mult,
        weight_decay=cfg.weight_decay,
        poly_power=cfg.poly_power,
        warmup_steps=cfg.warmup_steps,
        ckpt_path=cfg.ckpt_path,
        delta_weights=cfg.delta_weights,
        load_ckpt_class_head=cfg.load_ckpt_class_head,
    )


def lightning_from_dict(config: Dict[str, Any], network) -> EoMTLightning:
    """Helper to map a plain dict (e.g. YAML) to the Lightning module."""

    return build_da3_eomt_lightning(
        DA3EoMTTSLoRAConfig(network=network, **config)
    )


class DA3EoMTTSLoRALightning(MaskClassificationInstance):
    """Extension of the EoMT Lightning stack with optional 3D losses.

    This class reuses the full loss/metric/annealing pipeline from
    ``MaskClassificationInstance`` while providing hooks to incorporate 3D
    consistency losses during Stage-2 training. If ``gt_3d`` is absent in the
    batch the behaviour matches the original EoMT training loop.
    """

    def __init__(
        self,
        lambda_3d: float = 0.0,
        lambda_2d: float = 1.0,
        stage: str = "stage2_multiview_3d",
        freeze_backbone: bool = False,
        freeze_depth_ray_head: bool = False,
        train_lora_only: bool = False,
        log_lora_norms: bool = True,
        **kwargs: Any,
    ):
        self.lambda_3d = lambda_3d
        self.lambda_2d = lambda_2d
        self.stage = stage
        self.freeze_backbone = freeze_backbone
        self.freeze_depth_ray_head = freeze_depth_ray_head
        self.train_lora_only = train_lora_only
        self.log_lora_norms = log_lora_norms
        super().__init__(**kwargs)

        if self.stage.startswith("stage1"):
            self._freeze_for_stage1()

    def forward(self, imgs, **kwargs):
        """Forward images through the wrapped network with optional extras."""

        return self.network(imgs, **kwargs)

    def _compute_query_centroids(
        self, mask_logits: torch.Tensor, intrinsics: torch.Tensor, poses: torch.Tensor
    ) -> torch.Tensor:
        """Project soft masks to coarse 3D centroids using unit depth.

        Args:
            mask_logits: ``(B, Q, H, W)`` mask logits.
            intrinsics: ``(B, 3, 3)`` camera intrinsics.
            poses: ``(B, 4, 4)`` camera-to-world transforms.

        Returns:
            Tensor of shape ``(B, Q, 3)`` containing one centroid per query and
            per-view. Unit depth is used as a placeholder when dense depth is
            unavailable; the goal is to provide a consistent Stage-2 loss
            surface without breaking compatibility with the EoMT training loop.
        """

        b, q, h, w = mask_logits.shape
        device = mask_logits.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(grid_x)
        pixels = torch.stack([grid_x, grid_y, ones], dim=-1).view(1, 1, h * w, 3)
        probs = mask_logits.sigmoid().view(b, q, 1, h * w)

        intrinsics = intrinsics.view(b, 1, 3, 3)
        poses = poses.view(b, 1, 4, 4)
        intrinsics_inv = torch.inverse(intrinsics)
        cam_rays = pixels @ intrinsics_inv.transpose(-1, -2)
        cam_points = cam_rays  # unit depth
        cam_points = cam_points.transpose(-1, -2)
        cam_points_h = torch.cat(
            [cam_points, torch.ones_like(cam_points[:, :, :1, :])], dim=2
        )  # (B, Q, 4, HW)
        world_points = poses @ cam_points_h
        world_points = world_points[:, :, :3, :]
        weighted = (world_points * probs).sum(-1)
        norm = probs.sum(-1).clamp(min=1e-6)
        return weighted / norm

    def compute_3d_loss(
        self,
        mask_logits: torch.Tensor,
        intrinsics: torch.Tensor,
        poses: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Chamfer-like centroid consistency across views.

        A minimal Stage-2 loss encouraging the same query slot to map to
        consistent 3D locations across views. Uses placeholder unit-depth
        centroids to keep the loss finite even when dense depth supervision is
        unavailable.
        """

        if mask_logits.dim() != 4:
            return None
        b, v = intrinsics.shape[:2]
        q = mask_logits.shape[1]
        centroids = self._compute_query_centroids(
            mask_logits.view(b * v, q, *mask_logits.shape[-2:]),
            intrinsics.view(b * v, 3, 3),
            poses.view(b * v, 4, 4),
        )
        centroids = centroids.view(b, v, q, 3)
        loss = 0.0
        pairs = 0
        for i in range(v):
            for j in range(i + 1, v):
                diff = centroids[:, i] - centroids[:, j]
                loss = loss + diff.pow(2).sum(dim=-1).mean()
                pairs += 1
        if pairs == 0:
            return None
        return loss / pairs

    def training_step(self, batch, batch_idx):
        if isinstance(batch, MultiViewBatch):
            b, v, c, h, w = batch.images.shape
            flat_imgs = batch.images.view(b * v, c, h, w)
            flat_targets = [t for per_view in batch.panoptic for t in per_view]

            mask_logits_per_block, class_logits_per_block = self(
                flat_imgs, intrinsics=batch.intrinsics, poses=batch.poses
            )

            losses_all_blocks = {}
            for i, (mask_logits, class_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block))
            ):
                losses = self.criterion(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=flat_targets,
                )
                block_postfix = self.block_postfix(i)
                losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
                losses_all_blocks |= losses

            loss_2d = self.criterion.loss_total(losses_all_blocks, self.log)
            total = self.lambda_2d * loss_2d

            mask_logits_last = mask_logits_per_block[-1]
            loss_3d = self.compute_3d_loss(
                mask_logits_last, batch.intrinsics, batch.poses
            )
            if loss_3d is not None and self.lambda_3d > 0:
                total = total + self.lambda_3d * loss_3d
                self.log("loss_3d", loss_3d, on_step=True, prog_bar=True)
            self.log("loss_2d", loss_2d, on_step=True, prog_bar=True)
            self.log("loss_total", total, on_step=True, prog_bar=True)
            if self.log_lora_norms:
                self._log_lora_norms()
            return total

        base_loss = super().training_step((batch.images, batch.panoptic), batch_idx)
        if self.log_lora_norms:
            self._log_lora_norms()
        if hasattr(self.network, "attn_mask_probs"):
            self.log(
                "attn_mask_prob_mean",
                self.network.attn_mask_probs.mean(),
                on_step=True,
                prog_bar=False,
            )
        return base_loss

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._log_lora_config()

    def _freeze_for_stage1(self) -> None:
        """Freeze DA3 backbone and depth-related heads while keeping LoRA trainable."""

        for _, param in self.network.named_parameters():
            param.requires_grad = False

        segmentation_modules = []
        if hasattr(self.network, "encoder"):
            segmentation_modules.append(getattr(self.network.encoder, "seg_head", None))
            segmentation_modules.append(getattr(self.network.encoder, "seg_query_embed", None))
        segmentation_modules.append(getattr(self.network, "class_predictor", None))

        for module in segmentation_modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

        for name, param in self.network.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            elif any(keyword in name for keyword in ["seg_query", "seg_queries"]):
                param.requires_grad = True

    def _log_lora_norms(self) -> None:
        lora_a_norms = []
        lora_b_norms = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name and param is not None:
                lora_a_norms.append(param.detach().norm())
            elif "lora_B" in name and param is not None:
                lora_b_norms.append(param.detach().norm())

        if lora_a_norms:
            mean_a = torch.stack(lora_a_norms).mean()
            self.log("lora_A_norm", mean_a, on_step=True, prog_bar=False)
        if lora_b_norms:
            mean_b = torch.stack(lora_b_norms).mean()
            self.log("lora_B_norm", mean_b, on_step=True, prog_bar=False)

    def _log_lora_config(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        cfg = getattr(self.network, "tslora_cfg", None)
        if cfg:
            self.logger.experiment.config.update({"tslora": cfg}, allow_val_change=True)

