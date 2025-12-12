from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from panopticapi.evaluation import pq_compute
from panopticapi.utils import id2rgb
from PIL import Image

from third_party.eomt.training.lightning_module import (
    LightningModule as EoMTLightningModule,
)


def _build_segments_info(
    panoptic_pred: torch.Tensor, inverse_class_map: Dict[int, int]
) -> tuple[torch.Tensor, List[dict]]:
    class_map = panoptic_pred[..., 0].long()
    segment_map = panoptic_pred[..., 1].long()
    segments_info: List[dict] = []

    # #region agent log
    import json as _json_log; open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:_build_segments_info:entry", "message": "Build segments info called", "data": {"panoptic_shape": list(panoptic_pred.shape), "unique_segments": int(segment_map.unique().numel()), "inverse_map_size": len(inverse_class_map)}, "hypothesisId": "B", "timestamp": __import__("time").time()}) + "\n")
    # #endregion

    for segment_id in segment_map.unique():
        seg_id = int(segment_id.item())
        if seg_id < 0:
            continue
        mask = segment_map == seg_id
        if mask.sum() == 0:
            continue
        class_id = int(class_map[mask][0].item())
        category_id = inverse_class_map.get(class_id, class_id)
        segments_info.append(
            {
                "id": seg_id,
                "category_id": category_id,
                "iscrowd": 0,
                "area": int(mask.sum().item()),
                "isthing": int(category_id < 80),
            }
        )

    # #region agent log
    open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:_build_segments_info:exit", "message": "Build segments info result", "data": {"num_segments": len(segments_info), "segment_ids": [s["id"] for s in segments_info], "category_ids": [s["category_id"] for s in segments_info]}, "hypothesisId": "B", "timestamp": __import__("time").time()}) + "\n")
    # #endregion

    return segment_map, segments_info


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
    """Thin wrapper around panopticapi PQ computation using EoMT postprocessing."""

    def __init__(
        self,
        gt_json: str | os.PathLike,
        gt_folder: str | os.PathLike,
        inverse_class_map: Dict[int, int],
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
    ) -> None:
        self.gt_json = Path(gt_json)
        self.gt_folder = Path(gt_folder)
        self.inverse_class_map = inverse_class_map
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh

        self._tmpdir = Path(tempfile.mkdtemp(prefix="da3_panoptic_pred_"))
        self.pred_dir = self._tmpdir / "panoptic_preds"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.pred_json = self.pred_dir / "pred_panoptic.json"

        self._predictions: list[dict] = []

    def _gather_predictions(self) -> List[dict]:
        if not dist.is_available() or not dist.is_initialized():
            return list(self._predictions)

        world_size = dist.get_world_size()
        gathered: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gathered, self._predictions)

        merged: List[dict] = []
        for preds in gathered:
            merged.extend(preds)
        return merged

    def process_batch(
        self,
        model,
        targets: Iterable[dict],
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        stuff_classes: Sequence[int],
    ) -> None:
        # #region agent log
        import json as _json_log; open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:process_batch:entry", "message": "process_batch called", "data": {"mask_logits_shape": list(mask_logits.shape), "class_logits_shape": list(class_logits.shape), "num_targets": len(list(targets)) if hasattr(targets, '__len__') else "unknown"}, "hypothesisId": "A,D", "timestamp": __import__("time").time()}) + "\n")
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
            segment_map, segments_info = _build_segments_info(
                panoptic_pred, self.inverse_class_map
            )
            image_id = int(target.get("image_id", i))
            # 直接使用 image_id 生成标准的 COCO panoptic 格式文件名（12位补零）
            file_name = f"{image_id:012d}.png"

            # #region agent log
            open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:process_batch:prediction_added", "message": "Prediction added", "data": {"image_id": image_id, "file_name": file_name, "num_segments": len(segments_info)}, "hypothesisId": "D,E", "timestamp": __import__("time").time()}) + "\n")
            # #endregion

            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "segment_map": segment_map.cpu(),
                }
            )

    def compute(self, global_rank: int = 0) -> Tuple[float, float, float] | None:
        if global_rank != 0 and (not dist.is_available() or not dist.is_initialized()):
            return None

        all_predictions = self._gather_predictions()
        if global_rank != 0:
            return None

        # 🔧 添加验证：如果没有预测，返回 None
        if not all_predictions:
            print("[Warning] No predictions to evaluate, skipping PQ computation")
            return None

        # 🔧 添加调试信息：对比预测和 ground truth
        print(f"[DEBUG] Total predictions collected: {len(all_predictions)}")
        predicted_ids = sorted(set(p['image_id'] for p in all_predictions))
        print(f"[DEBUG] Number of unique image_ids: {len(predicted_ids)}")
        print(f"[DEBUG] First 10 predicted IDs: {predicted_ids[:10]}")
        print(f"[DEBUG] Last 10 predicted IDs: {predicted_ids[-10:]}")
        
        # 检查 ground truth 有多少图像
        with open(str(self.gt_json), 'r') as f:
            gt_data = json.load(f)
            gt_ids = sorted([img['id'] for img in gt_data.get('images', [])])
            print(f"[DEBUG] Ground truth total images: {len(gt_ids)}")
            missing_ids = sorted(set(gt_ids) - set(predicted_ids))
            if missing_ids:
                print(f"[DEBUG] Missing {len(missing_ids)} predictions")
                print(f"[DEBUG] First 20 missing IDs: {missing_ids[:20]}")
                print(f"[DEBUG] Is 3845 in missing? {3845 in missing_ids}")
            
            # #region agent log
            import json as _json_log
            open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": "coco_panoptic_evaluator.py:compute:gt_comparison", "message": "GT vs Predictions comparison", "data": {"gt_count": len(gt_ids), "pred_count": len(predicted_ids), "missing_count": len(missing_ids), "first_50_missing": missing_ids[:50], "extra_pred_ids": sorted(set(predicted_ids) - set(gt_ids))[:20]}, "hypothesisId": "F,H,I", "timestamp": __import__("time").time()}) + "\n")
            # #endregion

        # 过滤掉缺失预测的图像，仅对可用的预测与对应 GT 计算 PQ
        if missing_ids:
            gt_data["images"] = [img for img in gt_data.get("images", []) if img["id"] in predicted_ids]
            gt_data["annotations"] = [
                ann for ann in gt_data.get("annotations", []) if ann.get("image_id") in predicted_ids
            ]
            filtered_gt_json = self._tmpdir / "filtered_gt.json"
            with filtered_gt_json.open("w") as f:
                json.dump(gt_data, f)
            gt_json_for_eval = filtered_gt_json
        else:
            gt_json_for_eval = self.gt_json

        with self.pred_json.open("w") as f:
            json.dump(
                {
                    "annotations": [
                        {
                            "image_id": pred["image_id"],
                            "file_name": pred["file_name"],
                            "segments_info": pred["segments_info"],
                        }
                        for pred in all_predictions
                    ]
                },
                f,
            )

        for pred in all_predictions:
            out_path = self.pred_dir / pred["file_name"]
            Image.fromarray(id2rgb(pred["segment_map"].numpy())).save(out_path)

        try:
            pq_res = pq_compute(
                gt_json_file=str(gt_json_for_eval),
                pred_json_file=str(self.pred_json),
                gt_folder=str(self.gt_folder),
                pred_folder=str(self.pred_dir),
            )

            # 🔧 添加验证：检查返回结果的格式
            if not isinstance(pq_res, dict):
                print(f"[Warning] pq_compute returned unexpected type: {type(pq_res)}, expected dict")
                return None

            # 🔧 验证必需的键是否存在
            if "All" not in pq_res or "Things" not in pq_res or "Stuff" not in pq_res:
                print(f"[Warning] pq_compute result missing required keys. Available keys: {pq_res.keys()}")
                return None

            # 🔧 验证嵌套结构
            if not isinstance(pq_res["All"], dict) or "pq" not in pq_res["All"]:
                print(f"[Warning] pq_compute result has unexpected structure")
                return None

            return (
                pq_res["All"]["pq"],
                pq_res["Things"]["pq"],
                pq_res["Stuff"]["pq"],
            )
        except Exception as e:
            print(f"[Error] PQ computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reset(self) -> None:
        self._predictions.clear()
        if self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="da3_panoptic_pred_"))
        self.pred_dir = self._tmpdir / "panoptic_preds"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.pred_json = self.pred_dir / "pred_panoptic.json"
