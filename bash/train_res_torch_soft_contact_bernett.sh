#!/bin/bash
# Slurm launcher for Bernett residual model with torch contact attention and
# pre-contact positional enrichment.
# Submit with: sbatch bash/train_res_torch_soft_contact_bernett.sh

#SBATCH --job-name=torchsoft-bernett
#SBATCH -p singhlab-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_torch_soft_contact_%a.out
#SBATCH --error=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_torch_soft_contact_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu

set -euo pipefail

CONDA_BASE=${CONDA_BASE:-/opt/apps/rhel9/Anaconda3-2024.02}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dscript}"

PY=${PY:-/hpc/home/wl324/projects/tt3d/data_archive/env/dscript/bin/python}

TRAIN=${TRAIN:-/hpc/home/wl324/projects/tt3d/data_archive/bernett_train.tsv}
TEST=${TEST:-/hpc/home/wl324/projects/tt3d/data_archive/bernett_validation.tsv}
EMBEDDING=${EMBEDDING:-/hpc/home/wl324/projects/tt3d/data_archive/esm2/bernett}
OUTPUT_BASE=${OUTPUT_BASE:-/hpc/home/wl324/projects/tt3d/data/results/bernett_torch_soft_contact}

DEVICE=${DEVICE:-0}
SEED=${SEED:-612}
LR=${LR:-0.00025}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
DROPOUT=${DROPOUT:-0.5}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}

INPUT_DIM=${INPUT_DIM:-1280}
PROJECTION_DIM=${PROJECTION_DIM:-100}
HIDDEN_DIM=${HIDDEN_DIM:-50}
KERNEL_WIDTH=${KERNEL_WIDTH:-7}
POOL_WIDTH=${POOL_WIDTH:-9}
GRID_SIZE=${GRID_SIZE:-128}
CLASSIFIER_D_MODEL=${CLASSIFIER_D_MODEL:-64}
ATTN_POOL_SIZE=${ATTN_POOL_SIZE:-16}
ATTN_DROPOUT=${ATTN_DROPOUT:-0.1}
MAP_OUT_CHANNELS=${MAP_OUT_CHANNELS:-8}

EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0001}
NEGATIVE_CENTER_LOSS_WEIGHT=${NEGATIVE_CENTER_LOSS_WEIGHT:-0.05}
NEGATIVE_CENTER_LOSS_MIN_COUNT=${NEGATIVE_CENTER_LOSS_MIN_COUNT:-2}
MIXUP_WEIGHT=${MIXUP_WEIGHT:-0.0}
MIXUP_LAM_MIN=${MIXUP_LAM_MIN:-0.75}
MIXUP_LAM_MAX=${MIXUP_LAM_MAX:-0.95}
SOFT_TARGET_LAMBDA=${SOFT_TARGET_LAMBDA:-0.7}
SOFT_TARGET_EMA_DECAY=${SOFT_TARGET_EMA_DECAY:-0.999}
SOFT_TARGET_POSITIVE_ONLY=${SOFT_TARGET_POSITIVE_ONLY:-1}

RUN_NAME="bernett_torch_soft_contact_lr${LR}_wd${WEIGHT_DECAY}_dp${DROPOUT}_g${GRID_SIZE}_d${CLASSIFIER_D_MODEL}_hd${HIDDEN_DIM}_s${SEED}_ch${MAP_OUT_CHANNELS}_ad${ATTN_DROPOUT}_lw${NEGATIVE_CENTER_LOSS_WEIGHT}_mx${MIXUP_LAM_MIN}-${MIXUP_LAM_MAX}_mw${MIXUP_WEIGHT}_st${SOFT_TARGET_LAMBDA}_ema${SOFT_TARGET_EMA_DECAY}_stpo${SOFT_TARGET_POSITIVE_ONLY}"
OUTPUT_FOLDER="${OUTPUT_BASE}/${RUN_NAME}"
mkdir -p "${OUTPUT_FOLDER}"

if [ ! -d "${EMBEDDING}" ] && [ ! -f "${EMBEDDING}" ]; then
    echo "ERROR: Embedding path is missing: ${EMBEDDING}" >&2
    exit 1
fi

export WANDB_NAME="${RUN_NAME}"
export WANDB_TAGS="bernett,train,res,torch_soft_contact,contact_torch,soft_target"
export WANDB_RUN_GROUP="tt3d_backbone_torch_soft_contact_bernett"
export WANDB_JOB_TYPE="train"

SOFT_TARGET_POSITIVE_ONLY_FLAG=()
if [ "${SOFT_TARGET_POSITIVE_ONLY}" -eq 1 ]; then
  SOFT_TARGET_POSITIVE_ONLY_FLAG=(--soft-target-positive-only)
fi

"${PY}" -u -m dscript.commands.train_res_torch_soft_contact \
  --train "${TRAIN}" \
  --test "${TEST}" \
  --embedding "${EMBEDDING}" \
  --input-dim "${INPUT_DIM}" \
  --outfile "${OUTPUT_FOLDER}/results.log" \
  --save-prefix "${OUTPUT_FOLDER}/model" \
  --device "${DEVICE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --attn-pool-size "${ATTN_POOL_SIZE}" \
  --attn-dropout "${ATTN_DROPOUT}" \
  --map-out-channels "${MAP_OUT_CHANNELS}" \
  --num-epochs "${EPOCHS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --negative-center-loss-weight "${NEGATIVE_CENTER_LOSS_WEIGHT}" \
  --negative-center-loss-min-count "${NEGATIVE_CENTER_LOSS_MIN_COUNT}" \
  --mixup-weight "${MIXUP_WEIGHT}" \
  --mixup-lam-min "${MIXUP_LAM_MIN}" \
  --mixup-lam-max "${MIXUP_LAM_MAX}" \
  --soft-target-lambda "${SOFT_TARGET_LAMBDA}" \
  --soft-target-ema-decay "${SOFT_TARGET_EMA_DECAY}" \
  "${SOFT_TARGET_POSITIVE_ONLY_FLAG[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --pool-width "${POOL_WIDTH}" \
  --grid-size "${GRID_SIZE}" \
  --classifier-d-model "${CLASSIFIER_D_MODEL}" \
  --kernel-width "${KERNEL_WIDTH}" \
  --dropout-p "${DROPOUT}" \
  --projection-dim "${PROJECTION_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --log_wandb \
  --wandb-entity bergerlab-mit \
  --wandb-project tt3d_backbone
