import json
from pathlib import Path

import pytest
import torch
from PIL import Image

panopticapi = pytest.importorskip("panopticapi")
from panopticapi.utils import id2rgb  # type: ignore  # noqa: E402

from depth_anything_3.eval.coco_panoptic_evaluator import (
    DA3CocoPanopticEvaluator,
    _build_segments_info,
)


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


def test_pq_computation_ignores_missing_predictions(tmp_path: Path):
    """PQ should still compute when some GT images have no prediction."""

    gt_folder = tmp_path / "panoptic_val2017"
    gt_folder.mkdir()
    gt_json = tmp_path / "panoptic_val2017.json"

    # Two ground-truth images, but we will only predict for image 2.
    segment_map = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    Image.fromarray(id2rgb(segment_map.numpy())).save(gt_folder / "000000000001.png")
    Image.fromarray(id2rgb(segment_map.numpy())).save(gt_folder / "000000000002.png")

    categories = [
        {"id": 1, "name": "thing", "isthing": 1, "supercategory": "thing", "color": [0, 0, 0]},
    ]
    annotations = [
        {
            "image_id": 1,
            "file_name": "000000000001.png",
            "segments_info": [{"id": 1, "category_id": 1, "iscrowd": 0, "area": 4, "bbox": [0, 0, 2, 2]}],
            "id": 1,
        },
        {
            "image_id": 2,
            "file_name": "000000000002.png",
            "segments_info": [{"id": 1, "category_id": 1, "iscrowd": 0, "area": 4, "bbox": [0, 0, 2, 2]}],
            "id": 2,
        },
    ]
    images = [
        {"id": 1, "width": 2, "height": 2, "file_name": "000000000001.png"},
        {"id": 2, "width": 2, "height": 2, "file_name": "000000000002.png"},
    ]

    with gt_json.open("w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    evaluator = DA3CocoPanopticEvaluator(
        gt_json=str(gt_json),
        gt_folder=str(gt_folder),
        inverse_class_map={0: 1},
    )

    # Only provide prediction for image_id 2
    class_map = torch.zeros((2, 2), dtype=torch.int64)
    panoptic_pred = torch.stack([class_map, segment_map], dim=-1)
    pred_segment_map, pred_segments_info = _build_segments_info(panoptic_pred, evaluator.inverse_class_map)
    evaluator._predictions.append(
        {
            "image_id": 2,
            "file_name": "000000000002.png",
            "segments_info": pred_segments_info,
            "segment_map": pred_segment_map,
        }
    )

    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is not None, "PQ computation should run even with missing predictions"
    pq_all, pq_things, pq_stuff = pq_scores
    assert pq_all == 1.0
    assert pq_things == 1.0
    assert pq_stuff == 0.0
