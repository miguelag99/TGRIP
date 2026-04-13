#!/bin/bash
#SBATCH --job-name=tgrip-training
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:2
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=tgrip
PATH_TO_SOURCE_CODE=/raid/${USER}/workspace/TGRIP
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh
IMG_USERNAME=perception

if [ ! -f "$OUTPUT_SQSH" ]; then
  echo "Error: $OUTPUT_SQSH not found. Please first pull and convert the Docker image to SquashFS format."
  exit 1
else
  echo "Found $OUTPUT_SQSH."
fi

srun \
  --gpus=2 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/home/${IMG_USERNAME}/workspace,/raid/smontiel/Datasets/nuscenes:/home/${IMG_USERNAME}/Datasets/nuscenes" \
  --container-env=WANDB_API_KEY \
  --container-env=HF_TOKEN \
  bash -c '
    uv sync
    ulimit -n 65535
    uv run tgrip/train.py
    echo "Training completed!"
  '
