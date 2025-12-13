import pytest
import torch

from depth_anything_3.eval.coco_panoptic_evaluator import (
    DA3CocoPanopticEvaluator,
)


def test_pq_computation_basic():
    """测试基本的 PQ 计算"""
    num_classes = 3  # 2个实际类 + 1个void
    stuff_classes = [2]  # stuff类别
    
    evaluator = DA3CocoPanopticEvaluator(
        num_classes=num_classes,
        stuff_classes=stuff_classes,
        mask_thresh=0.8,
        overlap_thresh=0.8,
    )
    
    # 构造一个简单的 batch
    # GT: masks [B, N, H, W], labels [B, N], target_ids [B, N]
    targets = [
        {
            "image_id": 1,
            "masks": torch.tensor([
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]
            ], dtype=torch.float32),  # [2, 2, 4]
            "labels": torch.tensor([0, 1], dtype=torch.long),  # thing, stuff
            "target_ids": torch.tensor([1, 2], dtype=torch.long),
            "orig_size": (2, 4),
        }
    ]
    
    # 预测：完美匹配 GT
    # mask_logits: [B, Q, H, W], class_logits: [B, Q, C]
    mask_logits = torch.zeros(1, 2, 2, 4)
    mask_logits[0, 0, :, :2] = 10.0  # query 0 预测左半部分
    mask_logits[0, 1, :, 2:] = 10.0  # query 1 预测右半部分
    
    class_logits = torch.zeros(1, 2, num_classes)
    class_logits[0, 0, 0] = 10.0  # query 0 预测类别 0
    class_logits[0, 1, 1] = 10.0  # query 1 预测类别 1
    
    # 模拟 process_batch（需要 mock model）
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    
    evaluator.process_batch(
        model=mock_model,
        targets=targets,
        mask_logits=mask_logits,
        class_logits=class_logits,
        stuff_classes=stuff_classes,
    )
    
    # 计算 PQ
    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is not None
    pq_all, pq_things, pq_stuff = pq_scores
    
    # 完美预测应该接近 1.0（可能因为插值有微小差异）
    assert pq_all > 0.8, f"PQ too low: {pq_all}"
    assert pq_things > 0.8, f"PQ_things too low: {pq_things}"
    assert pq_stuff > 0.8, f"PQ_stuff too low: {pq_stuff}"
    
    evaluator.reset()
    assert len(evaluator._predictions) == 0


def test_process_batch_multiple_images():
    """测试处理多张图片"""
    evaluator = DA3CocoPanopticEvaluator(
        num_classes=3,
        stuff_classes=[2],
        mask_thresh=0.8,
        overlap_thresh=0.8,
    )
    
    # 构造 2 张图片的 batch
    targets = [
        {
            "image_id": 1,
            "masks": torch.randn(2, 4, 4) > 0,
            "labels": torch.tensor([0, 1]),
            "target_ids": torch.tensor([1, 2]),
            "orig_size": (4, 4),
        },
        {
            "image_id": 2,
            "masks": torch.randn(3, 4, 4) > 0,
            "labels": torch.tensor([0, 1, 2]),
            "target_ids": torch.tensor([1, 2, 3]),
            "orig_size": (4, 4),
        }
    ]
    
    mask_logits = torch.randn(2, 100, 4, 4)
    class_logits = torch.randn(2, 100, 3)
    
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    
    evaluator.process_batch(
        model=mock_model,
        targets=targets,
        mask_logits=mask_logits,
        class_logits=class_logits,
        stuff_classes=[2],
    )
    
    # 应该有 2 个预测
    assert len(evaluator._predictions) == 2
    assert evaluator._predictions[0]["image_id"] == 1
    assert evaluator._predictions[1]["image_id"] == 2
    
    # 计算不应该崩溃
    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is not None


def test_empty_predictions():
    """测试没有预测时的行为"""
    evaluator = DA3CocoPanopticEvaluator(
        num_classes=3,
        stuff_classes=[2],
    )
    
    pq_scores = evaluator.compute(global_rank=0)
    assert pq_scores is None  # 应该返回 None