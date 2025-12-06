#!/usr/bin/env bash
# Example command for Stage-1 single-view panoptic training on 8xA100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=da3_seg_coco

python -m depth_anything_3.training.main_seg_stage1 fit \
  --config configs/seg/stage1_da3b_coco640.yaml \
  --data.init_args.root /path/to/coco \
  --data.init_args.panoptic_json_train /path/to/panoptic_train2017.json \
  --data.init_args.panoptic_json_val /path/to/panoptic_val2017.json \
  --trainer.devices 8 \
  --trainer.accelerator gpu \
  --trainer.strategy ddp \
  --trainer.precision 16-mixed \
  --model.init_args.lr 0.0002 \
  --model.init_args.num_masked_layers 4 \
  --model.init_args.attn_mask_annealing_enabled true

