#!/bin/bash
#SBATCH --job-name=train
#SBATCH -p cellbio-dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-12
#SBATCH --mem=128G
#SBATCH --time=60:00:00
#SBATCH --output=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_train_%a.out
#SBATCH --error=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_train_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu

module purge
module load Anaconda3/2024.02

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /hpc/home/wl324/projects/env/dscript

PY=/hpc/home/wl324/projects/env/dscript/bin/python


$PY - <<'EOF'
import sys, torch
import torch_optimizer
print('torch_optimizer OK')
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
sys.exit(0 if ok else 1)
EOF


if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available. Exiting."
    exit 1
fi

export WANDB_PROJECT=tt3d_backbone
export OUTPUT_PREFIX="run_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p /hpc/home/wl324/projects/tt3d/data/outputs

export WANDB_TAGS="bernett,aug,tauri"
wandb agent bergerlab-mit/tt3d_backbone/5onopuko --count 1
