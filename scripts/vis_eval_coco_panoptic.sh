#!/usr/bin/env bash
# Example command to visualize COCO panoptic validation samples and compute PQ metrics.

set -euo pipefail

# Update these paths before running
CKPT_PATH="/home/jovyan/ybai_ws/Depth-Anything-3/checkpoints/stage1_da3b_coco640/epoch=epoch=19-step=step=29520-loss=train_loss=26.6661.ckpt"
CONFIG_PATH="/home/jovyan/ybai_ws/Depth-Anything-3/configs/seg/stage1_da3b_coco640.yaml"
SAVE_DIR="outputs/panoptic_vis2"
DEVICE="cuda:0"
NUM_SAMPLES=16

python -m tools.vis_and_eval_coco_panoptic \
  --ckpt "${CKPT_PATH}" \
  --config "${CONFIG_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --save-dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --mask-thresh 0.25 \
  --overlap-thresh 0.25 \
  --max-samples 16
