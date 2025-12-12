import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

panopticapi = pytest.importorskip("panopticapi")
from panopticapi.utils import id2rgb  # type: ignore  # noqa: E402

from depth_anything_3.eval.coco_panoptic_evaluator import (
    DA3CocoPanopticEvaluator,
    _build_segments_info,
)

# #region agent log
import json as _json_log
def _log(location: str, message: str, data: dict, hypothesis: str):
    open("/home/jovyan/ybai_ws/.cursor/debug.log", "a").write(_json_log.dumps({"location": f"test_coco_panoptic_pq.py:{location}", "message": message, "data": data, "hypothesisId": hypothesis, "timestamp": __import__("time").time()}) + "\n")
# #endregion


def test_pq_computation_matches_ground_truth(tmp_path: Path):
    gt_folder = tmp_path / "panoptic_val2017"
    gt_folder.mkdir()
    gt_json = tmp_path / "panoptic_val2017.json"

    # Build a simple 2x4 ground-truth segmentation with one thing and one stuff region.
    segment_map = torch.tensor(
        [[1, 1, 2, 2], [1, 1, 2, 2]], dtype=torch.int64
    )
    Image.fromarray(id2rgb(segment_map.numpy())).save(gt_folder / "000000000001.png")

    categories = [
        {"id": 1, "name": "thing", "isthing": 1, "supercategory": "thing", "color": [0, 0, 0]},
        {"id": 2, "name": "stuff", "isthing": 0, "supercategory": "stuff", "color": [10, 10, 10]},
    ]
    annotations = [
        {
            "image_id": 1,
            "file_name": "000000000001.png",
            "segments_info": [
                {"id": 1, "category_id": 1, "iscrowd": 0, "area": 4, "bbox": [0, 0, 2, 2]},
                {"id": 2, "category_id": 2, "iscrowd": 0, "area": 4, "bbox": [2, 0, 2, 2]},
            ],
            "id": 1,
        }
    ]
    images = [
        {
            "id": 1,
            "width": segment_map.shape[1],
            "height": segment_map.shape[0],
            "file_name": "000000000001.png",
        }
    ]

    with gt_json.open("w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    evaluator = DA3CocoPanopticEvaluator(
        gt_json=str(gt_json),
        gt_folder=str(gt_folder),
        inverse_class_map={0: 1, 1: 2},
    )

    class_map = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64
    )
    panoptic_pred = torch.stack([class_map, segment_map], dim=-1)
    pred_segment_map, pred_segments_info = _build_segments_info(panoptic_pred, evaluator.inverse_class_map)

    evaluator._predictions.append(
        {
            "image_id": 1,
            "file_name": "000000000001.png",
            "segments_info": pred_segments_info,
            "segment_map": pred_segment_map,
        }
    )

    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is not None
    pq_all, pq_things, pq_stuff = pq_scores
    for score in (pq_all, pq_things, pq_stuff):
        assert score == 1.0

    evaluator.reset()
    assert evaluator._predictions == []


def test_process_batch_coverage(tmp_path: Path):
    """测试 process_batch 方法，模拟 validation_step 的流程"""
    _log("test_process_batch_coverage:start", "Test started", {}, "C")
    
    gt_folder = tmp_path / "panoptic_val2017"
    gt_folder.mkdir()
    gt_json = tmp_path / "panoptic_val2017.json"

    # 创建 2x4 的 GT 分割图
    segment_map = torch.tensor(
        [[1, 1, 2, 2], [1, 1, 2, 2]], dtype=torch.int64
    )
    Image.fromarray(id2rgb(segment_map.numpy())).save(gt_folder / "000000000001.png")

    categories = [
        {"id": 1, "name": "thing", "isthing": 1, "supercategory": "thing", "color": [0, 0, 0]},
        {"id": 2, "name": "stuff", "isthing": 0, "supercategory": "stuff", "color": [10, 10, 10]},
    ]
    annotations = [
        {
            "image_id": 1,
            "file_name": "000000000001.png",
            "segments_info": [
                {"id": 1, "category_id": 1, "iscrowd": 0, "area": 4, "bbox": [0, 0, 2, 2]},
                {"id": 2, "category_id": 2, "iscrowd": 0, "area": 4, "bbox": [2, 0, 2, 2]},
            ],
            "id": 1,
        }
    ]
    images = [
        {
            "id": 1,
            "width": segment_map.shape[1],
            "height": segment_map.shape[0],
            "file_name": "000000000001.png",
        }
    ]

    with gt_json.open("w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    evaluator = DA3CocoPanopticEvaluator(
        gt_json=str(gt_json),
        gt_folder=str(gt_folder),
        inverse_class_map={0: 1, 1: 2},
    )

    # 创建 mock model，模拟 to_per_pixel_preds_panoptic 的返回值
    mock_model = MagicMock()
    # to_per_pixel_preds_panoptic 返回格式：[H, W, 2]，第一通道是 class_id，第二通道是 segment_id
    class_map = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
    panoptic_pred = torch.stack([class_map, segment_map], dim=-1)
    
    # 模拟 EoMTLightningModule.to_per_pixel_preds_panoptic 返回
    from unittest.mock import patch
    with patch("depth_anything_3.eval.coco_panoptic_evaluator.EoMTLightningModule.to_per_pixel_preds_panoptic") as mock_preds:
        mock_preds.return_value = [panoptic_pred]
        
        # 创建假的 mask_logits 和 class_logits
        # mask_logits: [B, Q, H, W]，class_logits: [B, Q, C]
        mask_logits = torch.randn(1, 100, 2, 4)
        class_logits = torch.randn(1, 100, 133)
        
        targets = [{"image_id": 1, "orig_size": (2, 4)}]
        
        _log("test_process_batch_coverage:before_process", "Calling process_batch", 
             {"mask_shape": list(mask_logits.shape), "class_shape": list(class_logits.shape)}, "C")
        
        evaluator.process_batch(
            model=mock_model,
            targets=targets,
            mask_logits=mask_logits,
            class_logits=class_logits,
            stuff_classes=[2],
        )
        
        _log("test_process_batch_coverage:after_process", "process_batch completed", 
             {"num_predictions": len(evaluator._predictions)}, "C")
    
    # 验证预测被正确添加
    assert len(evaluator._predictions) == 1
    pred = evaluator._predictions[0]
    assert pred["image_id"] == 1
    assert pred["file_name"] == "000000000001.png"
    assert len(pred["segments_info"]) == 2
    
    _log("test_process_batch_coverage:computing_pq", "Computing PQ scores", {}, "C")
    
    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is not None
    pq_all, pq_things, pq_stuff = pq_scores
    
    _log("test_process_batch_coverage:end", "Test completed", 
         {"pq_all": pq_all, "pq_things": pq_things, "pq_stuff": pq_stuff}, "C")
    
    for score in (pq_all, pq_things, pq_stuff):
        assert score == 1.0
