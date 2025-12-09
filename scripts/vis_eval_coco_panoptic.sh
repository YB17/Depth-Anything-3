#!/usr/bin/env bash
# Example command to visualize COCO panoptic validation samples and compute PQ metrics.

set -euo pipefail

# Update these paths before running
CKPT_PATH="/path/to/your_checkpoint.ckpt"
CONFIG_PATH="configs/seg_panoptic/your_config.yaml"
SAVE_DIR="outputs/panoptic_vis"
DEVICE="cuda:0"
NUM_SAMPLES=16

python tools/vis_and_eval_coco_panoptic.py \
  --ckpt "${CKPT_PATH}" \
  --config "${CONFIG_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --save-dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --mask-thresh 0.8 \
  --overlap-thresh 0.8
