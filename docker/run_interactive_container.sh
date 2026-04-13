#!/bin/bash

# Please run this file directly, not with sbatch

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=tgrip
PATH_TO_SOURCE_CODE=/raid/${USER}/workspace/TGRIP
IMAGE_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh
IMG_USERNAME=perception

srun \
  --gpus=1 \
  --partition=H200 \
  --time=00:30:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --job-name=diffpred-bash \
  --container-env=WANDB_API_KEY \
  --container-image="$IMAGE_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/home/${IMG_USERNAME}/workspace,/raid/smontiel/Datasets/nuscenes:/home/${IMG_USERNAME}/Datasets/nuscenes" \
  --pty bash
