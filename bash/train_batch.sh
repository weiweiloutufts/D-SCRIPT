#!/bin/bash
#SBATCH --job-name=train_batch40_5
#SBATCH -p cellbio-dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_train_batch_%a.out
#SBATCH --error=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_train_batch_%a.err
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

lam=0.99
lr=0.0001
wd=0.0005

TOPSY_TURVY=
TRAIN=/hpc/home/wl324/projects/tt3d/data_archive/bernett_train.tsv
TEST=/hpc/home/wl324/projects/tt3d/data_archive/bernett_validation.tsv

# These are *paths* and numeric dims, not full flags
EMBEDDING=/hpc/home/wl324/projects/tt3d/data_archive/esm2_bernett
EMBEDDING_DIM=1280

OUTPUT_BASE=/hpc/home/wl324/projects/tt3d/data/results
OUTPUT_FOLDER=${OUTPUT_BASE}/bernett_esm2_train_batch_5_5_lr${lr}_lam${lam}_wd${wd}
OUTPUT_PREFIX=bernett
FOLDSEEK_FASTA=/hpc/home/wl324/tt3d/p_tt3d_new/fasta/full_fasta.fasta

DEVICE=0

usage() {
    echo "Usage: ./train.sh [-d DEVICE] [-v] [-f] [-F foldseek_fasta_file] [-e EMB_DIR] [-E EMB_DIM]

    -d: CUDA device index (default: 0)
    -v: When set, trains a Topsy-Turvy model
    -f: When set, trains a TT3D model (enables foldseek features)
    -F: Foldseek 3di sequences fasta file (used only when -f is set)
    -e: Embedding h5py file or directory (default: $EMBEDDING)
    -E: Embedding dimension (default: $EMBEDDING_DIM)
    "
}

while getopts "d:t:fF:T:e:E:vo:p:" args; do
    case $args in
        d) DEVICE=${OPTARG} ;;
        t) TRAIN=${OPTARG} ;;
        T) TEST=${OPTARG} ;;
        e) EMBEDDING=${OPTARG} ;;
        E) EMBEDDING_DIM=${OPTARG} ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925" ;;
        o) OUTPUT_FOLDER=${OUTPUT_BASE}/${OPTARG} ;;
        p) OUTPUT_PREFIX=${OPTARG} ;;
        f) FOLDSEEK="" ;;
        F) FOLDSEEK_FASTA=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done

# Build flags from the path / dim variables
EMBEDDING_FLAG="--embedding ${EMBEDDING}"
EMBEDDING_DIM_FLAG="--input-dim ${EMBEDDING_DIM}"

BACKBONE_CMD=""
if [ -n "$BACKBONE" ]; then
    BACKBONE_CMD="--allow_backbone3di --backbone3di_fasta ${FOLDSEEK_FASTA}"
fi

FOLDSEEK_CMD=""
if [ -n "$FOLDSEEK" ]; then
    FOLDSEEK_CMD="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA}" # --add_foldseek_after_projection"
fi

if [ ! -d "${OUTPUT_FOLDER}" ]; then
    mkdir -p "${OUTPUT_FOLDER}"
fi



export WANDB_NAME="bernett_batch_lr${lr}_lam${lam}_wd${wd}"
export WANDB_TAGS="bernett,aug,tauri,batch"
export WANDB_RUN_GROUP="tt3d_backbone_batch"
export WANDB_JOB_TYPE="train_batch"

$PY -m dscript.commands.train_batch \
  --train "${TRAIN}" \
  --test "${TEST}" \
  ${EMBEDDING_FLAG} \
  ${EMBEDDING_DIM_FLAG} \
  ${TOPSY_TURVY} \
  --outfile "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_results.log" \
  --save-prefix "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}" \
  --device "${DEVICE}" \
  --lr "${lr}" \
  --lambda "${lam}" \
  --num-epoch 20 \
  --weight-decay "${wd}" \
  --batch-size 5 \
  --pool-width 9 \
  --kernel-width 7 \
  --dropout-p 0.2 \
  --projection-dim 100 \
  --hidden-dim 50 \
  ${BACKBONE_CMD} \
  ${FOLDSEEK_CMD} \
   --log_wandb \
   --wandb-entity bergerlab-mit \
   --wandb-project tt3d_backbone

