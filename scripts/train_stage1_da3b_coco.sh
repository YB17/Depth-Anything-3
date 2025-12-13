#!/usr/bin/env bash
# Example command for Stage-1 single-view panoptic training on 8xA100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=da3_seg_coco
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1



torchrun --standalone --nproc_per_node=2 \
  -m depth_anything_3.training.main_seg_stage1 fit \
  --config configs/seg/stage1_da3b_coco640.yaml \
  --data.root /home/jovyan/ybai_ws/dataset/dataset/coco \
  --data.panoptic_json_train /home/jovyan/ybai_ws/dataset/dataset/coco/annotations/panoptic_train2017.json \
  --data.panoptic_json_val /home/jovyan/ybai_ws/dataset/dataset/coco/annotations/panoptic_val2017.json \
  --trainer.devices 2 \
  --trainer.accelerator gpu \
  --trainer.strategy ddp_find_unused_parameters_true \
  --trainer.precision 16-mixed \
  --model.lr 0.0004 \
  --model.num_masked_layers 4 \
  --model.attn_mask_annealing_enabled false \
  --model.da3_pretrained_path=/cache/model/da3/ \
  # --model.enable_dino_teacher true \
  # --model.dino_teacher_ckpt /cache/model/dinov3/dinov3_vitb16_pretrain.pth \
#  --resume_from_weights /cache/model/stage1_da3b_coco640/last.ckpt

