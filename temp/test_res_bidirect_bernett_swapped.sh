#!/bin/bash
#SBATCH --job-name=test_res_bidirect_swap
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_test_res_bidirect_swap.out
#SBATCH --error=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_test_res_bidirect_swap.err

set -euo pipefail

PY=/hpc/home/wl324/projects/tt3d/data_archive/env/dscript/bin/python
MODEL=/hpc/home/wl324/projects/tt3d/data/results/bernett_esm2_train_res_bidirect_lr0.00025_wd0.0002_dp0.5_grid32_d8_sym1.0/bernett_res_bidirect_best_model.sav
EMBEDDING_DIR=/hpc/home/wl324/projects/tt3d/data_archive/esm2/bernett
TEST=/hpc/home/wl324/D-SCRIPT/temp/bernett_test_swapped.tsv
OP_FILE=/hpc/home/wl324/D-SCRIPT/temp/bernett_res_bidirect_swapped

"$PY" - <<'PYEOF'
import sys
import torch

ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
sys.exit(0 if ok else 1)
PYEOF

"$PY" -u -m dscript.commands.evaluate_res_bidirect \
    --model "$MODEL" \
    --embeddings "$EMBEDDING_DIR" \
    --test "$TEST" \
    -d 0 \
    -o "$OP_FILE"
