#!/usr/bin/env bash
# Example command for Stage-1 single-view panoptic training on 8xA100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=da3_seg_coco

python -m depth_anything_3.training.main_seg_stage1 fit \
  --config configs/seg/stage1_da3b_coco640.yaml \
  --data.root /home/jovyan/ybai_ws/dataset/dataset/coco \
  --data.panoptic_json_train /home/jovyan/ybai_ws/dataset/dataset/coco/panoptic_train2017.json \
  --data.panoptic_json_val /home/jovyan/ybai_ws/dataset/dataset/coco/panoptic_val2017.json \
  --trainer.devices 2 \
  --trainer.accelerator gpu \
  --trainer.strategy ddp \
  --trainer.precision 16-mixed \
  --model.lr 0.0002 \
  --model.num_masked_layers 4 \
  --model.attn_mask_annealing_enabled true

