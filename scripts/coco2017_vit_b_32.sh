#!/bin/bash

DIR_ROOT="/home/cs-shin1/zutis"
P_CONFIG="${DIR_ROOT}/configs/coco2017_val_imagenet_pass_n500_vit_b_32.yaml"

if [ "$#" -eq  "0" ]
then
  CUDA_LAUNCH_BLOCKING=1 python3 "${DIR_ROOT}/main.py" \
  --p_config "${P_CONFIG}"
else
  CUDA_LAUNCH_BLOCKING=1 python3 "${DIR_ROOT}/main.py" \
  --p_config "${P_CONFIG}" \
  --p_state_dict "$1"
fi