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
            segment_map, segments_info = _build_segments_info(
                panoptic_pred, self.inverse_class_map
            )

            image_id = int(target.get("image_id", i))
            file_name = str(target.get("file_name", f"{image_id}.png"))
            if not file_name.endswith(".png"):
                file_name = file_name.replace(".jpg", ".png")

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

        with self.pred_json.open("w") as f:
            json.dump(
                [
                    {
                        "image_id": pred["image_id"],
                        "file_name": pred["file_name"],
                        "segments_info": pred["segments_info"],
                    }
                    for pred in all_predictions
                ],
                f,
            )

        for pred in all_predictions:
            out_path = self.pred_dir / pred["file_name"]
            Image.fromarray(id2rgb(pred["segment_map"].numpy())).save(out_path)

        pq_res = pq_compute(
            gt_json_file=str(self.gt_json),
            pred_json_file=str(self.pred_json),
            gt_folder=str(self.gt_folder),
            pred_folder=str(self.pred_dir),
        )

        return (
            pq_res["All"]["pq"],
            pq_res["Things"]["pq"],
            pq_res["Stuff"]["pq"],
        )

    def reset(self) -> None:
        self._predictions.clear()
        if self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="da3_panoptic_pred_"))
        self.pred_dir = self._tmpdir / "panoptic_preds"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.pred_json = self.pred_dir / "pred_panoptic.json"
