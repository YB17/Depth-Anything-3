from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from lightning.pytorch.utilities import rank_zero_only
import json
import os

from third_party.eomt.training.mask_classification_loss import MaskClassificationLoss
from third_party.eomt.training.two_stage_warmup_poly_schedule import (
    TwoStageWarmupPolySchedule,
)
from third_party.dinov3.loss.gram_loss import GramLoss

from depth_anything_3.model.segmentation.head_eomt_adapter import EoMTSegHead

from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.teacher.dinov3_teacher import DINOv3Teacher
from depth_anything_3.utils.checkpoint_utils import (
    load_da3_pretrained_backbone,
    resolve_da3_ckpt_path,
)
from depth_anything_3.eval import DA3CocoPanopticEvaluator

from third_party.eomt.datasets.coco_panoptic_directory import CLASS_MAPPING

class DA3SegPanopticModule(pl.LightningModule):
    """Stage-1 LightningModule for DA3 + segmentation branch panoptic training."""

    def __init__(
        self,
        network_config: dict,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        da3_pretrained_path: str = "",
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
        enable_panoptic_eval: bool = True,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
    ) -> None:
        super().__init__()
        # # 保存参数为实例变量
        # self.enable_dino_teacher = enable_dino_teacher
        # self.dino_teacher_ckpt = dino_teacher_ckpt
        # self.dino_teacher_alpha = dino_teacher_alpha
        
        self.network = self._build_network(network_config)
        self.img_size = img_size
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.enable_panoptic_eval = enable_panoptic_eval
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.mask_annealing_poly_factor = mask_annealing_poly_factor
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.poly_power = poly_power

        self.save_hyperparameters(ignore=["network"])


        # 🔧 简化 teacher 配置获取
        # 优先从 network_config 读取，然后尝试从 hparams 读取（命令行覆盖）
        self.enable_dino_teacher = network_config.get("enable_dino_teacher", False)
        self.dino_teacher_ckpt = network_config.get("dino_teacher_ckpt", "")
        self.gram_layers = network_config.get("gram_layers", [5, 7, 9, 11])
        self.lambda_gram = float(network_config.get("lambda_gram", 1.0))
        
        # 🔧 如果命令行有覆盖，使用命令行的值
        if hasattr(self.hparams, "enable_dino_teacher"):
            self.enable_dino_teacher = self.hparams.enable_dino_teacher
        if hasattr(self.hparams, "dino_teacher_ckpt"):
            self.dino_teacher_ckpt = self.hparams.dino_teacher_ckpt
        if hasattr(self.hparams, "gram_layers"):
            self.gram_layers = self.hparams.gram_layers
        if hasattr(self.hparams, "lambda_gram"):
            self.lambda_gram = self.hparams.lambda_gram

        raw_ckpt_path = getattr(self.hparams, "da3_pretrained_path", "")
        if hasattr(self.hparams, "model"):
            raw_ckpt_path = getattr(self.hparams.model, "da3_pretrained_path", raw_ckpt_path)

        ckpt_path = resolve_da3_ckpt_path(raw_ckpt_path or "")
        if ckpt_path:
            # DinoV2 内部的 pretrained 属性才是实际的 DinoVisionTransformer
            # checkpoint 的键对应的是 DinoVisionTransformer 的结构
            target_backbone = getattr(self.network.backbone, 'pretrained', self.network.backbone)
            rank_zero_only(load_da3_pretrained_backbone)(
                target_backbone, ckpt_path, strict=False
            )

        self.num_masked_layers = num_masked_layers
        if self.num_masked_layers is None:
            self.num_masked_layers = getattr(self.network.backbone, "num_seg_masked_layers", 0)

        embed_dim = getattr(self.network.backbone, "embed_dim", None)
        patch_grid = None
        if hasattr(self.network.backbone, "patch_embed"):
            patch_grid = getattr(self.network.backbone.patch_embed, "patches_resolution", None)
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

        if self.enable_dino_teacher:
            self.dino_teacher = DINOv3Teacher(
                ckpt_path=self.dino_teacher_ckpt,
                img_size=592,
                layers=self.gram_layers,
                patch_size=16,
            )
            self.gram_loss = GramLoss()
        else:
            self.dino_teacher = None
            self.gram_loss = None

        if self.dino_teacher is not None:
            self.dino_teacher.eval()
        # self.print(
        #     f"[DINOv3 teacher] enable={self.enable_dino_teacher}, gram_layers={self.gram_layers}, lambda_gram={self.lambda_gram}"
        # )
        
        # ✅ 在这里调用，确保所有组件都已创建
        self._freeze_pretrained_components()

        # ✅ 只保存标志位，不立即初始化 evaluator
        self.panoptic_evaluator: DA3CocoPanopticEvaluator | None = None
        # 不要在这里初始化！

    def setup(self, stage: str) -> None:
        """Lightning hook: 在训练/验证开始前调用，此时 hparams 已完全填充"""
        # 🔧 添加调试输出
        print(f"[DEBUG] setup() called with stage={stage}")
        print(f"[DEBUG] enable_panoptic_eval={self.enable_panoptic_eval}")
        print(f"[DEBUG] panoptic_evaluator is None: {self.panoptic_evaluator is None}")
        
        # 🔧 修复：在 fit 或 validate 时都初始化
        if stage in ("fit", "validate") and self.enable_panoptic_eval and self.panoptic_evaluator is None:
            print(f"[DEBUG] Starting evaluator initialization...")
            
            # 🔧 关键修复：从 trainer 的 datamodule 获取配置
            gt_json = ""
            data_root = ""
            
            if self.trainer and hasattr(self.trainer, "datamodule"):
                datamodule = self.trainer.datamodule
                # 直接从 datamodule 的初始化参数获取
                gt_json = getattr(datamodule, "panoptic_json_val", "")
                data_root = getattr(datamodule, "path", "") or getattr(datamodule, "root", "")
                print(f"[DEBUG] Got from datamodule - gt_json: {gt_json}, data_root: {data_root}")
            
            # 如果 datamodule 没有，尝试从 hparams 获取（fallback）
            if not gt_json:
                gt_json = getattr(self.hparams.data, "panoptic_json_val", "") if hasattr(self.hparams, "data") else ""
                data_root = getattr(self.hparams.data, "root", "") if hasattr(self.hparams, "data") else ""
                gt_json = gt_json or getattr(self.hparams, "panoptic_json_val", "")
                print(f"[DEBUG] Got from hparams - gt_json: {gt_json}, data_root: {data_root}")
            
            # 🔧 添加验证：如果路径为空，禁用 evaluator
            if not gt_json:
                print("[Warning] panoptic_json_val not configured, disabling PQ evaluation")
                print(f"[DEBUG] hasattr(self.hparams, 'data'): {hasattr(self.hparams, 'data')}")
                if hasattr(self.hparams, "data"):
                    print(f"[DEBUG] self.hparams.data: {self.hparams.data}")
                return
            
            if gt_json and not os.path.isabs(gt_json):
                gt_json = os.path.join(data_root, gt_json)
            
            # 验证文件是否存在
            if not os.path.exists(gt_json):
                print(f"[Error] GT JSON file not found: {gt_json}")
                return
            
            gt_folder = os.path.join(os.path.dirname(os.path.dirname(gt_json)), "panoptic_val2017")
            
            # 验证文件夹是否存在
            if not os.path.exists(gt_folder):
                print(f"[Error] GT folder not found: {gt_folder}")
                return
            
            try:
                _, thing_ids, stuff_ids = self._build_coco_categories(gt_json)  # 传入 gt_json
            except FileNotFoundError as e:
                print(f"[Warning] Failed to build categories: {e}")
                thing_ids, stuff_ids = [], []

            inverse_class_map = {v: k for k, v in CLASS_MAPPING.items()}
            self.panoptic_evaluator = DA3CocoPanopticEvaluator(
                gt_json=gt_json,
                gt_folder=gt_folder,
                inverse_class_map=inverse_class_map,
                mask_thresh=self.mask_thresh,
                overlap_thresh=self.overlap_thresh,
            )
            # 🔧 添加确认日志
            from lightning.pytorch.utilities import rank_zero_info
            rank_zero_info(f"✅ [PQ Evaluator] Initialized successfully")
            rank_zero_info(f"   GT JSON: {gt_json}")
            rank_zero_info(f"   GT Folder: {gt_folder}")

    def _freeze_pretrained_components(self):
        """冻结DA3 backbone和depth head，只训练segmentation相关组件"""
        
        # 1. 冻结整个backbone（除了segmentation相关部分）
        backbone = self.network.backbone
        if hasattr(backbone, 'pretrained'):
            backbone = backbone.pretrained
        
        for name, param in backbone.named_parameters():
            # 只有seg_相关的参数保持可训练
            if 'seg_tokens' in name or 'seg_blocks' in name or 'seg_adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 2. 冻结depth head (DualDPT)
        if hasattr(self.network, 'head') and self.network.head is not None:
            for param in self.network.head.parameters():
                param.requires_grad = False
        
        # 3. 确保seg_head可训练
        for param in self.seg_head.parameters():
            param.requires_grad = True
        
        # 4. 打印冻结状态（调试用）
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = total_params - trainable_params
        
        from lightning.pytorch.utilities import rank_zero_info
        rank_zero_info(f"📌 Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        rank_zero_info(f"🔥 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        rank_zero_info(f"📊 Total parameters: {total_params:,}")

    def _build_network(self, config: dict) -> nn.Module:
        """从配置字典构建网络"""
        # 构建backbone (net)
        net_config = config.get('net', {})
        net = DinoV2(
            name=net_config.get('name', 'vitb'),
            out_layers=net_config.get('out_layers', [5, 7, 9, 11]),
            alt_start=net_config.get('alt_start', 4),
            qknorm_start=net_config.get('qknorm_start', 4),
            rope_start=net_config.get('rope_start', 4),
            cat_token=net_config.get('cat_token', True),
            seg_cfg=net_config.get('seg_cfg'),
        )
        
        # 构建head
        head_config = config.get('head', {})
        head = DualDPT(
            dim_in=head_config.get('dim_in', 1536),
            output_dim=head_config.get('output_dim', 2),
            features=head_config.get('features', 128),
            out_channels=head_config.get('out_channels', [96, 192, 384, 768]),
        )
        
        # 构建完整网络
        return DepthAnything3Net(net=net, head=head)

    def _build_coco_categories(self, gt_json: str = "") -> Tuple[List[dict], List[int], List[int]]:
        """
        从 COCO panoptic JSON 文件中提取 categories, thing_ids, stuff_ids
        
        Args:
            gt_json: panoptic JSON 文件的路径。如果不提供，会尝试从 hparams 获取。
        """
        # 如果没有传入 gt_json，尝试从 hparams 获取（fallback）
        if not gt_json:
            gt_json = getattr(self.hparams.data, "panoptic_json_val", "") if hasattr(self.hparams, "data") else ""
            data_root = getattr(self.hparams.data, "root", "") if hasattr(self.hparams, "data") else ""
            gt_json = gt_json or getattr(self.hparams, "panoptic_json_val", "")
            if gt_json and not os.path.isabs(gt_json):
                gt_json = os.path.join(data_root, gt_json)

        if not gt_json or not os.path.exists(gt_json):
            raise FileNotFoundError(f"Cannot locate panoptic json at {gt_json}")

        with open(gt_json, "r") as f:
            data = json.load(f)

        categories: List[dict] = data.get("categories", [])
        thing_ids, stuff_ids = [], []
        for cat in categories:
            is_thing = bool(cat.get("isthing", cat.get("isthing", cat.get("isthing", 0))))
            if not is_thing and cat.get("id", 0) < 80:
                is_thing = True

            if is_thing:
                thing_ids.append(cat["id"])
            else:
                stuff_ids.append(cat["id"])

        if not thing_ids and not stuff_ids:
            thing_ids = [cid for cid in CLASS_MAPPING if cid < 80]
            stuff_ids = [cid for cid in CLASS_MAPPING if cid >= 80]

        return categories, thing_ids, stuff_ids

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

    def get_mask_prob(self, layer_idx: int, global_step: int) -> float:
        if not self.training or not self.attn_mask_annealing_enabled or layer_idx >= self.num_masked_layers:
            return 0.0

        step = global_step + layer_idx
        start = self.attn_mask_annealing_start_steps[layer_idx]
        end = self.attn_mask_annealing_end_steps[layer_idx]
        if step <= start:
            return 1.0
        if step >= end:
            return 0.0
        t = (step - start) / max(1, end - start)
        return float((1.0 - t) ** self.mask_annealing_poly_factor)

    def build_seg_attn_mask(self, mask_logits: torch.Tensor, prob: float, num_heads: int) -> Optional[torch.Tensor]:
        if (not self.training) or prob <= 0.0:
            return None
        if torch.rand((), device=mask_logits.device) >= prob:
            return None

        # 🔧 关键修复：下采样mask到patch分辨率
        # mask_logits形状: [B, Q, upscaled_H, upscaled_W]
        # 需要下采样到 [B, Q, patch_H, patch_W] = [B, Q, 37, 37]
        import torch.nn.functional as F
        patch_h, patch_w = self.img_size[0] // 14, self.img_size[1] // 14  # 518 // 14 = 37
        
        # 使用adaptive_avg_pool2d下采样到patch分辨率
        mask_logits_downsampled = F.adaptive_avg_pool2d(
            mask_logits, 
            output_size=(patch_h, patch_w)
        )
        
        # reshape: [B, Q, patch_h, patch_w] -> [B, Q, patch_h * patch_w]
        allowed = mask_logits_downsampled.reshape(
            mask_logits_downsampled.shape[0], 
            mask_logits_downsampled.shape[1], 
            -1
        ) > 0
        
        # 注意：seg_tokens包含1个cls_token + patch_tokens
        # 需要为cls_token添加一个always-allowed的位置
        cls_allowed = torch.ones(
            allowed.shape[0], 
            allowed.shape[1], 
            1,  # 为cls_token添加1个位置
            dtype=allowed.dtype, 
            device=allowed.device
        )
        allowed = torch.cat([cls_allowed, allowed], dim=2)  # [B, Q, 1 + patch_h*patch_w]
        
        attn_mask = (~allowed).repeat_interleave(num_heads, dim=0)
        return attn_mask

    def _extract_seg_tokens(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, dict):
            return output.get("seg_tokens", {})
        return getattr(output, "seg_tokens", {})

    def forward(
        self,
        imgs: torch.Tensor,
        seg_attn_mask_fn=None,
        seg_head_fn=None,
        apply_seg_head_to_intermediate: bool = True,
        apply_seg_head_to_last: bool = True,
    ):
        return self.network(
            imgs,
            seg_attn_mask_fn=seg_attn_mask_fn,
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

    def _create_log_wrapper(self, prefix: str):
        """创建带前缀的log包装函数"""
        def log_fn(name, value, **kwargs):
            # 确保使用正确的前缀
            if name.startswith("losses/train_"):
                name = name.replace("losses/train_", f"losses/{prefix}_")
            elif not name.startswith("losses/"):
                pass  # 保持原样
            self.log(name, value, **kwargs)
        return log_fn

    def training_step(self, batch: Any, batch_idx: int):
        imgs, targets = batch
        seg_attn_mask_fn = None
        if self.training:
            seg_attn_mask_fn = lambda mask_logits, layer_idx, num_heads: self.build_seg_attn_mask(  # noqa: E731
                mask_logits=mask_logits,
                prob=self.get_mask_prob(layer_idx, self.global_step),
                num_heads=num_heads,
            )
        network_out = self(
            imgs,
            seg_attn_mask_fn=seg_attn_mask_fn,
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

        loss_seg = self.criterion.loss_total(
            losses_all_blocks,
            self._create_log_wrapper("train")  # ← 使用train前缀
        )
        loss_total = loss_seg
        loss_gram = torch.tensor(0.0, device=loss_seg.device)

        if self.enable_dino_teacher and self.gram_loss is not None and self.current_epoch < 2:
            student_tokens = seg_tokens.get("distill_tokens", {})
            if student_tokens:
                imgs_input = imgs
                while imgs_input.dim() > 4:
                    imgs_input = imgs_input.squeeze(1)  # 移除所有单维度
                
                # Resize 到 DINOv3 的输入尺寸
                imgs_teacher = F.interpolate(
                    imgs_input,
                    size=(592, 592),
                    mode="bilinear",
                    align_corners=False,
                )
                with torch.no_grad():
                    teacher_feats = self.dino_teacher(imgs_teacher)

                for l in self.gram_layers:
                    if l not in student_tokens or l not in teacher_feats:
                        continue
                    z_t = student_tokens[l]
                    t_l = teacher_feats[l]
                    assert (
                        z_t.shape[1] == t_l.shape[1]
                    ), f"Mismatch in tokens: student={z_t.shape[1]}, teacher={t_l.shape[1]}"
                    # b_s, t_s, c = z_t.shape
                    # h = w = int(t_s**0.5)
                    # z_t_map = z_t.view(b_s, h, w, c).permute(0, 3, 1, 2).contiguous()
                    # t_l_map = t_l.view(b_s, h, w, c).permute(0, 3, 1, 2).contiguous()
                    loss_gram = loss_gram + self.gram_loss(z_t, t_l)
                if len(self.gram_layers) > 0:
                    loss_gram = loss_gram / len(self.gram_layers)
                loss_total = loss_total + self.lambda_gram * loss_gram

        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("losses/train_seg", loss_seg, prog_bar=False, sync_dist=True)
        if self.enable_dino_teacher:
            self.log("losses/train_gram", loss_gram, prog_bar=False, sync_dist=True)

        mask_probs = [self.get_mask_prob(i, self.global_step) for i in range(self.num_masked_layers)]
        for i, prob in enumerate(mask_probs):
            self.log(f"anneal/p_mask_layer_{i}", prob, prog_bar=False)
        return loss_total

    def validation_step(self, batch: Any, batch_idx: int):
        # 🔧 添加调试输出
        if batch_idx == 0:
            print(f"[DEBUG] validation_step called! evaluator is None: {self.panoptic_evaluator is None}")
        
        imgs, targets = batch
        # #region agent log
        import json as _json_log
        _image_ids = [int(t.get("image_id", -1)) for t in targets]
        open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "seg_panoptic_module.py:validation_step:entry", "message": "Batch received", "data": {"batch_idx": batch_idx, "batch_size": len(targets), "image_ids": _image_ids}, "hypothesisId": "F,G", "timestamp": __import__("time").time()}) + "\n")
        # #endregion
        network_out = self(
            imgs,
            seg_attn_mask_fn=None,
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

        total_loss = self.criterion.loss_total(
            losses_all_blocks,
            self._create_log_wrapper("val")  # ← 使用val前缀
        )
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)

        if self.panoptic_evaluator is not None and head_outputs:
            with torch.no_grad():
                preds = head_outputs[-1]
                self.panoptic_evaluator.process_batch(
                    model=self,
                    targets=targets,
                    mask_logits=preds["pred_masks"],
                    class_logits=preds["pred_logits"],
                    stuff_classes=self.stuff_classes,
                )
        else:
            # #region agent log
            import json as _json_log
            _skip_reason = "evaluator_none" if self.panoptic_evaluator is None else "head_outputs_empty"
            open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "seg_panoptic_module.py:validation_step:skipped", "message": "process_batch skipped", "data": {"batch_idx": batch_idx, "skip_reason": _skip_reason, "head_outputs_len": len(head_outputs) if head_outputs else 0}, "hypothesisId": "G,J", "timestamp": __import__("time").time()}) + "\n")
            # #endregion

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1,},
        }

    def on_validation_epoch_end(self) -> None:
        if self.panoptic_evaluator is None:
            return

        # #region agent log
        import json as _json_log
        _pred_count = len(self.panoptic_evaluator._predictions)
        _pred_ids = sorted(set(p['image_id'] for p in self.panoptic_evaluator._predictions))
        open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "seg_panoptic_module.py:on_validation_epoch_end:before_compute", "message": "Before compute", "data": {"num_predictions": _pred_count, "num_unique_ids": len(_pred_ids), "first_10_ids": _pred_ids[:10], "last_10_ids": _pred_ids[-10:]}, "hypothesisId": "F,H", "timestamp": __import__("time").time()}) + "\n")
        # #endregion

        try:
            pq_scores = self.panoptic_evaluator.compute(global_rank=self.trainer.global_rank)
            if pq_scores is None:
                return

            pq_all, pq_things, pq_stuff = pq_scores
            self.log("val_PQ", pq_all, prog_bar=True, sync_dist=False, on_step=False, on_epoch=True)
            self.log("val_PQ_things", pq_things, sync_dist=False, on_step=False, on_epoch=True)
            self.log("val_PQ_stuff", pq_stuff, sync_dist=False, on_step=False, on_epoch=True)
            
            from lightning.pytorch.utilities import rank_zero_info
            rank_zero_info(f"📊 Epoch {self.current_epoch} PQ: {pq_all:.4f} (Things: {pq_things:.4f}, Stuff: {pq_stuff:.4f})")
        except Exception as e:
            from lightning.pytorch.utilities import rank_zero_warn
            rank_zero_warn(f"⚠️  PQ computation failed: {e}")
        finally:
            self.panoptic_evaluator.reset()

