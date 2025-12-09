from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from omegaconf import OmegaConf
from panopticapi.evaluation import pq_compute
from panopticapi.utils import id2rgb
from PIL import Image
import torch
import torch.nn.functional as F

from depth_anything_3.training.seg_panoptic_module import DA3SegPanopticModule
from depth_anything_3.utils.build_seg_panoptic_module import build_model_and_datamodule_from_config
from third_party.eomt.datasets.coco_panoptic_directory import CLASS_MAPPING
from third_party.eomt.training.lightning_module import LightningModule as EoMTLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and evaluate COCO panoptic predictions")
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--config", required=True, help="Path to Hydra config used for training")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of validation samples to visualize")
    parser.add_argument("--save-dir", default="outputs/panoptic_vis", help="Directory to store visualizations and predictions")
    parser.add_argument("--device", default="cuda:0", help="Computation device (e.g., cuda:0 or cpu)")
    parser.add_argument("--mask-thresh", type=float, default=0.8, help="Mask confidence threshold")
    parser.add_argument("--overlap-thresh", type=float, default=0.8, help="Minimum overlap ratio for valid segments")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    return parser.parse_args()


def build_inverse_class_mapping() -> Dict[int, int]:
    return {v: k for k, v in CLASS_MAPPING.items()}


def load_model_and_data(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config)
    _, datamodule = build_model_and_datamodule_from_config(cfg)

    model_kwargs = OmegaConf.to_container(getattr(cfg, "model", {}), resolve=True) or {}
    model = DA3SegPanopticModule.load_from_checkpoint(
        args.ckpt,
        strict=False,
        **model_kwargs,
    )
    model.to(args.device)
    model.eval()

    datamodule.setup("fit")
    return model, datamodule, cfg


def panoptic_from_logits(
    model: DA3SegPanopticModule,
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


def postprocess_batch(
    model: DA3SegPanopticModule,
    batch,
    stuff_classes: Sequence[int],
    mask_thresh: float,
    overlap_thresh: float,
) -> List[dict]:
    imgs, targets = batch
    imgs = imgs.to(model.device)

    network_out = model(
        imgs,
        seg_attn_mask_fn=None,
        seg_head_fn=model.seg_head,
        apply_seg_head_to_intermediate=False,
        apply_seg_head_to_last=True,
    )
    seg_tokens = model._extract_seg_tokens(network_out)
    head_outputs = model._collect_head_outputs(seg_tokens, compute_if_missing=True)
    if not head_outputs:
        return []

    preds = head_outputs[-1]
    mask_logits = preds["pred_masks"]
    class_logits = preds["pred_logits"]

    results = []
    for i in range(mask_logits.shape[0]):
        target = targets[i]
        orig_size = target.get("orig_size", mask_logits.shape[-2:])
        if isinstance(orig_size, torch.Tensor):
            orig_size = tuple(int(v) for v in orig_size)
        mask_logits_resized = F.interpolate(
            mask_logits[i][None], size=orig_size, mode="bilinear", align_corners=False
        )[0]
        class_logits_i = class_logits[i].unsqueeze(0)
        panoptic_pred = panoptic_from_logits(
            model,
            mask_logits_resized,
            class_logits_i,
            stuff_classes=stuff_classes,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
        )
        results.append(
            {
                "panoptic": panoptic_pred.cpu(),
                "target": target,
                "orig_size": orig_size,
            }
        )
    return results


def build_segments_info(panoptic_pred: torch.Tensor, inverse_class_map: Dict[int, int]) -> tuple[np.ndarray, List[dict]]:
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

    return segment_map.cpu().numpy().astype(np.int64), segments_info


def load_image_for_vis(datamodule, target: dict) -> Image.Image:
    img_path = Path(datamodule.path) / "val2017" / target.get("file_name", "")
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    img = target.get("image_tensor")
    if img is not None:
        return Image.fromarray(img)
    raise FileNotFoundError(f"Could not locate image for target {target}")


def visualize_samples(
    model: DA3SegPanopticModule,
    datamodule,
    device: str,
    save_dir: Path,
    num_samples: int,
    stuff_classes: Sequence[int],
    inverse_class_map: Dict[int, int],
    mask_thresh: float,
    overlap_thresh: float,
) -> None:
    if num_samples <= 0:
        return

    random.seed(42)
    vis_dir = save_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    val_loader = datamodule.val_dataloader()
    saved = 0
    for batch in val_loader:
        imgs, targets = batch
        imgs_np = (imgs[:, 0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for target, img_np in zip(targets, imgs_np):
            target["image_tensor"] = img_np

        batch = (imgs.to(device), targets)
        batch_results = postprocess_batch(
            model,
            batch,
            stuff_classes=stuff_classes,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
        )
        for item in batch_results:
            panoptic_pred = item["panoptic"]
            target = item["target"]
            segment_map, _ = build_segments_info(panoptic_pred, inverse_class_map)

            orig_img = load_image_for_vis(datamodule, target)
            if orig_img.size[::-1] != segment_map.shape:
                orig_img = orig_img.resize((segment_map.shape[1], segment_map.shape[0]), Image.BILINEAR)
            color_seg = id2rgb(segment_map)
            overlay = (
                0.5 * np.asarray(orig_img).astype(np.float32)
                + 0.5 * color_seg.astype(np.float32)
            ).astype(np.uint8)

            combined = np.concatenate([np.asarray(orig_img), overlay], axis=1)
            image_id = target.get("image_id", "unknown")
            out_path = vis_dir / f"{image_id}_panoptic.png"
            Image.fromarray(combined).save(out_path)

            saved += 1
            if saved >= num_samples:
                return


def generate_predictions(
    model: DA3SegPanopticModule,
    datamodule,
    save_dir: Path,
    stuff_classes: Sequence[int],
    inverse_class_map: Dict[int, int],
    mask_thresh: float,
    overlap_thresh: float,
    max_samples: int = None,
) -> tuple[Path, Path]:
    pred_dir = save_dir / "panoptic_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_json = pred_dir / "pred_panoptic.json"

    predictions = []
    val_loader = datamodule.val_dataloader()
    processed = 0
    for batch in val_loader:
        batch_results = postprocess_batch(
            model,
            batch,
            stuff_classes=stuff_classes,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
        )
        for item in batch_results:
            panoptic_pred = item["panoptic"]
            target = item["target"]
            segment_map, segments_info = build_segments_info(panoptic_pred, inverse_class_map)

            image_id = int(target.get("image_id", 0))
            file_name = f"{str(target.get('file_name', image_id)).replace('.jpg', '').replace('.png', '')}.png"
            Image.fromarray(id2rgb(segment_map)).save(pred_dir / file_name)
            predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                }
            )
            processed += 1

        if max_samples is not None and processed >= max_samples:  # ← 外层也检查
            break

    with pred_json.open("w") as f:
        json.dump(predictions, f)

    return pred_json, pred_dir


def compute_pq(cfg, pred_json: Path, pred_dir: Path) -> dict:
    data_cfg = OmegaConf.to_container(getattr(cfg, "data", {}), resolve=True) or {}
    gt_json = Path(data_cfg.get("panoptic_json_val", ""))
    if not gt_json.is_absolute():
        gt_json = Path(data_cfg.get("root", "")) / gt_json

    if gt_json.name.endswith(".json"):
        gt_folder = gt_json.parent.parent / "panoptic_val2017"
    else:
        gt_folder = Path(data_cfg.get("root", "")) / "panoptic_val2017"

    pq_res = pq_compute(
        gt_json_file=str(gt_json),
        pred_json_file=str(pred_json),
        gt_folder=str(gt_folder),
        pred_folder=str(pred_dir),
    )
    print(f"PQ_all:    {pq_res['All']['pq']}")
    print(f"PQ_things: {pq_res['Things']['pq']}")
    print(f"PQ_stuff:  {pq_res['Stuff']['pq']}")
    return pq_res


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, datamodule, cfg = load_model_and_data(args)
    stuff_classes = getattr(model, "stuff_classes", [])
    inverse_class_map = build_inverse_class_mapping()

    visualize_samples(
        model,
        datamodule,
        device=args.device,
        save_dir=save_dir,
        num_samples=args.num_samples,
        stuff_classes=stuff_classes,
        inverse_class_map=inverse_class_map,
        mask_thresh=args.mask_thresh,
        overlap_thresh=args.overlap_thresh,
    )

    pred_json, pred_dir = generate_predictions(
        model,
        datamodule,
        save_dir=save_dir,
        stuff_classes=stuff_classes,
        inverse_class_map=inverse_class_map,
        mask_thresh=args.mask_thresh,
        overlap_thresh=args.overlap_thresh,
        max_samples=args.max_samples,
    )

    compute_pq(cfg, pred_json, pred_dir)


if __name__ == "__main__":
    main()
