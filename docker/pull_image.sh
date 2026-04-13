#!/bin/bash
#SBATCH --job-name=tgrip-install
#SBATCH --partition=H200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=tgrip
PATH_TO_SOURCE_CODE=/raid/${USER}/workspace/TGRIP
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh
IMG_USERNAME=perception

if [ ! -d "/raid/${USER}/enroot/sqsh/" ]; then
  mkdir -p "/raid/${USER}/enroot/sqsh/"
fi

if [ ! -f "$OUTPUT_SQSH" ]; then
  enroot import \
    -o "$OUTPUT_SQSH" \
    docker://miguelag99/${REPO_NAME}:v1
else
  echo "Found $OUTPUT_SQSH, skipping import."
fi

echo "SBATCH cpus-per-task: $SLURM_CPUS_PER_TASK"


srun \
  --gpus=1 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/home/${IMG_USERNAME}/workspace,/raid/smontiel/Datasets/nuscenes:/home/${IMG_USERNAME}/Datasets/nuscenes" \
  bash -c "
    ls -l /home/${IMG_USERNAME}/workspace
    cd /home/${IMG_USERNAME}/workspace
    echo "🔄 Checking virtual environment..."
    if [ ! -d ".venv" ]; then
      echo " .venv not found — updating virtual environment..."

      # Run uv sync and capture output
      if ! uv sync --link-mode=copy; then
        echo "❌ uv sync failed, cleaning up..."
        rm -rf .venv
        exit 1
      fi

      echo "✅ .venv updated"
    else
      echo "✅ .venv found"
    fi
    echo "Installation completed!"
  "
