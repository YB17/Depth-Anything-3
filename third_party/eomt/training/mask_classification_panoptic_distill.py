# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import copy
import logging

import torch
import torch.nn.functional as F

from training.mask_classification_panoptic import MaskClassificationPanoptic
from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule


class MaskClassificationPanopticDistill(MaskClassificationPanoptic):
    def __init__(
        self,
        network,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        distill: Optional[dict] = None,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            stuff_classes=stuff_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.distill_cfg = distill or {}
        self.lambda_feat = self.distill_cfg.get("lambda_feat", 0.0)
        self.lambda_logit = self.distill_cfg.get("lambda_logit", 0.0)
        self.distill_temp = self.distill_cfg.get("temperature", 1.0)
        self.feature_layers = self.distill_cfg.get("feature_layers", None)
        self.student_pretrained_path = self.distill_cfg.get("student_pretrained_path")

        if (
            self.student_pretrained_path
            and hasattr(self.network, "encoder")
            and hasattr(self.network.encoder, "_load_backbone_ckpt")
        ):
            try:
                self.network.encoder._load_backbone_ckpt(self.student_pretrained_path)
            except Exception as exc:
                logging.warning(
                    f"Failed to load student backbone weights from "
                    f"{self.student_pretrained_path}: {exc}"
                )

        teacher_ckpt_path = self.distill_cfg.get("teacher_ckpt_path")
        self.teacher_model = self._build_teacher_model(self.network, teacher_ckpt_path)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        student_outputs = self(
            imgs,
            return_backbone_feats=True,
            return_seg_layers=True,
        )

        panoptic_outputs = student_outputs["panoptic"]
        loss_gt, _ = self.compute_panoptic_loss(
            panoptic_outputs[0], panoptic_outputs[1], targets
        )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                imgs / 255.0,
                return_backbone_feats=True,
                return_seg_layers=True,
            )

        loss_feat = self._compute_feature_distill_loss(
            student_outputs["backbone_feats"],
            teacher_outputs["backbone_feats"],
        )

        loss_cls_kl, loss_mask_bce = self._compute_logit_distill_loss(
            student_outputs["seg_layers"][-1],
            teacher_outputs["seg_layers"][-1],
        )

        loss_distill = (
            self.lambda_feat * loss_feat
            + self.lambda_logit * (loss_cls_kl + loss_mask_bce)
        )
        loss_total = loss_gt + loss_distill

        self.log(
            "train_loss_gt",
            loss_gt,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_feat",
            loss_feat,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_cls_kl",
            loss_cls_kl,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_mask_bce",
            loss_mask_bce,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_distill",
            loss_distill,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_total",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss_total

    def configure_optimizers(self):
        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.encoder.backbone.blocks)
        block_i = backbone_blocks

        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        for name, param in reversed(list(self.named_parameters())):
            if not param.requires_grad or name.startswith("teacher_model"):
                continue

            lr = self.lr

            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")

                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True

                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)

                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.lr

                if (
                    is_block
                    and (block_i in l2_blocks)
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _compute_feature_distill_loss(self, stu_feats, tea_feats):
        if not stu_feats or not tea_feats:
            return torch.tensor(0.0, device=self.device)

        layers = (
            self.feature_layers
            if self.feature_layers is not None
            else list(range(min(len(stu_feats), len(tea_feats))))
        )

        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for idx in layers:
            if idx >= len(stu_feats) or idx >= len(tea_feats):
                continue
            f_s = stu_feats[idx]
            f_t = tea_feats[idx].to(f_s.device)

            f_s_n = F.normalize(f_s, dim=-1)
            f_t_n = F.normalize(f_t, dim=-1)
            loss = loss + F.mse_loss(f_s_n, f_t_n)
            count += 1

        if count > 0:
            loss = loss / count

        return loss

    def _compute_logit_distill_loss(self, stu_last, tea_last):
        s_logits = stu_last["pred_logits"]
        t_logits = tea_last["pred_logits"].to(s_logits.device)

        tau = self.distill_temp

        t_prob = F.softmax(t_logits / tau, dim=-1)
        s_log_prob = F.log_softmax(s_logits / tau, dim=-1)
        loss_cls_kl = F.kl_div(
            s_log_prob,
            t_prob,
            reduction="batchmean",
        ) * (tau**2)

        s_masks = stu_last["pred_masks"]
        t_masks = tea_last["pred_masks"].to(s_masks.device)

        if t_masks.shape[-2:] != s_masks.shape[-2:]:
            t_masks = F.interpolate(
                t_masks,
                size=s_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        t_mask_prob = t_masks.sigmoid()
        loss_mask_bce = F.binary_cross_entropy_with_logits(
            s_masks,
            t_mask_prob,
        )

        return loss_cls_kl, loss_mask_bce

    def _build_teacher_model(self, network, ckpt_path: Optional[str]):
        teacher = copy.deepcopy(network)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            clean_state = {}
            teacher_state = teacher.state_dict()
            for key, value in ckpt.items():
                clean_key = key
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key.replace("_orig_mod.", "", 1)
                if clean_key.startswith("network."):
                    clean_key = clean_key.replace("network.", "", 1)

                if clean_key in teacher_state:
                    clean_state[clean_key] = value

            incompatible = teacher.load_state_dict(clean_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                logging.info(
                    f"Teacher checkpoint loaded with missing keys {incompatible.missing_keys} "
                    f"and unexpected keys {incompatible.unexpected_keys}"
                )

        for p in teacher.parameters():
            p.requires_grad = False

        teacher.eval()
        return teacher
