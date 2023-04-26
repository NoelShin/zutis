#!/bin/bash

DIR_ROOT="/home/cs-shin1/zutis"
DIR_DATASET="/home/cs-shin1/datasets/coco"

ARCH="vit_b_16"
#P_STATE_DICT="${DIR_ROOT}/pretrained_weights/coco_${ARCH}.pt"
P_CONFIG="${DIR_ROOT}/configs/coco2017_val_imagenet_pass_n500_${ARCH}.yaml"

CUDA_LAUNCH_BLOCKING=1 python3 "${DIR_ROOT}/coco20k_eval.py" \
--dir_dataset "${DIR_DATASET}" \
--dir_ckpt "${DIR_ROOT}/ckpt" \
--p_state_dict "$1" \
--p_config "${P_CONFIG}" \
--nms_type "hard"
