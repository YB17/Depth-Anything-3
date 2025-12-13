from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchmetrics.detection import PanopticQuality
from torchmetrics.functional.detection._panoptic_quality_common import (
    _calculate_iou,
    _get_color_areas,
    _prepocess_inputs,
)

from third_party.eomt.training.lightning_module import (
    LightningModule as EoMTLightningModule,
)


def _panoptic_from_logits(
    model,
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    stuff_classes: Sequence[int],
    mask_thresh: float,
    overlap_thresh: float,
) -> torch.Tensor:
    preds = EoMTLightningModule.to_per_pixel_preds_panoptic(
        model,
        [mask_logits],
        class_logits,
        stuff_classes,
        mask_thresh,
        overlap_thresh,
    )
    return preds[0]


class DA3CocoPanopticEvaluator:
    """Compute PQ metrics using EoMT-style accumulation instead of panopticapi."""

    def __init__(
        self,
        num_classes: int,
        stuff_classes: Sequence[int],
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
    ) -> None:
        # ✅ 添加验证
        if not stuff_classes:
            raise ValueError("stuff_classes cannot be empty")
        
        if any(cls_id < 0 or cls_id >= num_classes for cls_id in stuff_classes):
            raise ValueError(
                f"All stuff_classes must be in range [0, {num_classes}), "
                f"but got: {[c for c in stuff_classes if c < 0 or c >= num_classes]}"
            )

        self.num_classes = num_classes
        self.stuff_classes = list(stuff_classes)
        self.thing_classes = [
            cls_id for cls_id in range(num_classes) if cls_id not in self.stuff_classes
        ]
        self.void_class = num_classes
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh

        self.metric = PanopticQuality(
            self.thing_classes,
            self.stuff_classes + [self.void_class],
            return_sq_and_rq=True,
            return_per_class=True,
        )

        self._predictions: list[dict] = []

        # ✅ 初始化日志
        try:
            from lightning.pytorch.utilities import rank_zero_info
            rank_zero_info(
                f"📊 [PQ Evaluator] Initialized: "
                f"{len(self.thing_classes)} things, "
                f"{len(self.stuff_classes)} stuff, "
                f"void={self.void_class}"
            )
        except:
            pass  # 不在Lightning环境中时忽略

    @classmethod
    def from_coco_standard(
        cls,
        num_classes: int = 133,
        thing_classes_end: int = 80,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
    ) -> "DA3CocoPanopticEvaluator":
        """
        使用 COCO 标准约定创建 evaluator
        
        Args:
            num_classes: 总类别数（默认133）
            thing_classes_end: thing类别结束位置（默认80）
            mask_thresh: mask阈值
            overlap_thresh: overlap阈值
        """
        stuff_classes = list(range(thing_classes_end, num_classes))
        return cls(
            num_classes=num_classes,
            stuff_classes=stuff_classes,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
        )

    def _build_target_panoptic(
        self, target: dict, spatial_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[int, bool]]:
        device = target["masks"].device
        h, w = spatial_size

        class_map = torch.full(
            (h, w), fill_value=self.void_class, device=device, dtype=torch.long
        )
        instance_map = torch.zeros((h, w), device=device, dtype=torch.long)

        masks = target["masks"].to(device=device)
        labels = target["labels"].to(device=device)
        target_ids = target.get("target_ids")
        if target_ids is None:
            target_ids = torch.arange(len(labels), device=device)
        target_ids = target_ids.to(device=device)

        if masks.ndim == 4:
            masks = masks[:, 0]

        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(h, w), mode="nearest")
            masks = masks[:, 0] > 0.5

        crowd_flags: Dict[int, bool] = {}
        is_crowd = target.get("is_crowd")
        if is_crowd is not None:
            is_crowd = is_crowd.to(device=device)
            for seg_id, flag in zip(target_ids.tolist(), is_crowd.tolist()):
                crowd_flags[int(seg_id)] = bool(flag)

        for mask, label, seg_id in zip(masks, labels, target_ids):
            class_map[mask] = label.long()
            instance_map[mask] = seg_id.long()

        panoptic = torch.stack((class_map, instance_map), dim=-1)
        return panoptic, crowd_flags

    def _update_panoptic_metric(
        self,
        panoptic_pred: torch.Tensor,
        panoptic_target: torch.Tensor,
        crowd_flags: Dict[int, bool],
    ) -> None:
        metric = self.metric

        # ✅ 确保 metric 在正确的设备上（修复设备不匹配问题）
        device = panoptic_pred.device
        if metric.iou_sum.device != device:
            metric.to(device)
            
        flatten_pred = _prepocess_inputs(
            metric.things,
            metric.stuffs,
            panoptic_pred[None, ...],
            metric.void_color,
            metric.allow_unknown_preds_category,
        )[0]
        flatten_target = _prepocess_inputs(
            metric.things,
            metric.stuffs,
            panoptic_target[None, ...],
            metric.void_color,
            True,
        )[0]

        pred_areas = _get_color_areas(flatten_pred)
        target_areas = _get_color_areas(flatten_target)
        intersection_matrix = torch.transpose(
            torch.stack((flatten_pred, flatten_target), -1), -1, -2
        )
        intersection_areas = _get_color_areas(intersection_matrix)

        pred_segment_matched = set()
        target_segment_matched = set()

        for pred_color, target_color in intersection_areas:
            if pred_color[0] == metric.void_color or target_color[0] == metric.void_color:
                continue

            if pred_color not in pred_areas or target_color not in target_areas:
                continue

            # ✅ 跳过类别不匹配的情况（关键修复！）
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

            if iou > 0.5:
                pred_segment_matched.add(pred_color)
                target_segment_matched.add(target_color)
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                metric.iou_sum[continuous_id] += iou
                metric.true_positives[continuous_id] += 1

        false_negative_colors = set(target_areas) - target_segment_matched
        false_positive_colors = set(pred_areas) - pred_segment_matched

        false_negative_colors.discard(metric.void_color)
        false_positive_colors.discard(metric.void_color)

        for target_color in list(false_negative_colors):
            void_target_area = intersection_areas.get((metric.void_color, target_color), 0)
            if void_target_area / target_areas[target_color] > 0.5:
                false_negative_colors.discard(target_color)

        crowd_by_cat_id: Dict[int, int] = {}
        for false_negative_color in false_negative_colors:
            if crowd_flags.get(false_negative_color[1], False):
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

    def process_batch(
        self,
        model,
        targets: Iterable[dict],
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        stuff_classes: Sequence[int],
    ) -> None:
        # #region agent log
        import json as _json_log

        open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:process_batch:entry", "message": "process_batch called", "data": {"mask_logits_shape": list(mask_logits.shape), "class_logits_shape": list(class_logits.shape), "num_targets": len(list(targets)) if hasattr(targets, '__len__') else "unknown"}, "hypothesisId": "A,D", "timestamp": __import__("time").time()}) + "\n")
        # #endregion
        for i, target in enumerate(targets):
            orig_size = target.get("orig_size", mask_logits.shape[-2:])
            if isinstance(orig_size, torch.Tensor):
                orig_size = tuple(int(v) for v in orig_size)
            mask_logits_resized = F.interpolate(
                mask_logits[i][None], size=orig_size, mode="bilinear", align_corners=False
            )[0]
            class_logits_i = class_logits[i].unsqueeze(0)
            panoptic_pred = _panoptic_from_logits(
                model,
                mask_logits_resized,
                class_logits_i,
                stuff_classes=stuff_classes,
                mask_thresh=self.mask_thresh,
                overlap_thresh=self.overlap_thresh,
            )
            # #region agent log
            open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:process_batch:after_panoptic", "message": "panoptic_from_logits result", "data": {"panoptic_pred_shape": list(panoptic_pred.shape), "image_id": int(target.get("image_id", i))}, "hypothesisId": "A", "timestamp": __import__("time").time()}) + "\n")
            # #endregion

            panoptic_target, crowd_flags = self._build_target_panoptic(
                target, panoptic_pred.shape[:2]
            )

            self._update_panoptic_metric(panoptic_pred, panoptic_target, crowd_flags)

            image_id = int(target.get("image_id", i))
            self._predictions.append({"image_id": image_id})

    def compute(self, global_rank: int = 0) -> Tuple[float, float, float] | None:
        if global_rank != 0 and (not dist.is_available() or not dist.is_initialized()):
            return None

        if not self._predictions:
            print("[Warning] No predictions to evaluate, skipping PQ computation")
            return None

        # ✅ 添加统计日志
        try:
            from lightning.pytorch.utilities import rank_zero_info
            rank_zero_info(f"📊 Computing PQ for {len(self._predictions)} images")
        except:
            pass

        tp = self.metric.true_positives.to(torch.float32)
        fp = self.metric.false_positives.to(torch.float32)
        fn = self.metric.false_negatives.to(torch.float32)
        iou_sum = self.metric.iou_sum.to(torch.float32)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tp)
            dist.all_reduce(fp)
            dist.all_reduce(fn)
            dist.all_reduce(iou_sum)

        # ✅ 添加统计信息
        try:
            from lightning.pytorch.utilities import rank_zero_info
            rank_zero_info(
                f"   Stats - TP: {tp.sum():.0f}, FP: {fp.sum():.0f}, FN: {fn.sum():.0f}"
            )
        except:
            pass
            
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
        pq_values, sq_values, rq_values = (
            result[:, 0],
            result[:, 1],
            result[:, 2],
        )

        num_things = len(self.metric.things)
        pq_things = pq_values[:num_things]
        pq_stuff = pq_values[num_things:]

        return (
            float(pq_values.mean()),
            float(pq_things.mean()) if len(pq_things) > 0 else 0.0,
            float(pq_stuff.mean()) if len(pq_stuff) > 0 else 0.0,
        )

    def reset(self) -> None:
        self._predictions.clear()
        self.metric.reset()
