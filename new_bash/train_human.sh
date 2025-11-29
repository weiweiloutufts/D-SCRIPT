#!/bin/bash
#SBATCH --job-name=train_human
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_train_%a.out
#SBATCH --error=logs/%A_train_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
module purge
module load ngc/1.0
module load anaconda/2024.10
module load pytorch/2.5.1-cuda12.1-cudnn9


cd /cluster/tufts/cowenlab/wlou01/D-SCRIPT
conda activate dscript  

TOPSY_TURVY=
TRAIN=/cluster/tufts/cowenlab/tt3d+/data/pair_files/human_train.mapped.tsv
TEST=/cluster/tufts/cowenlab/tt3d+/data/pair_files/human_test.mapped.tsv

# These are *paths* and numeric dims, not full flags
EMBEDDING=/cluster/tufts/cowenlab/tt3d+/data/esm2_mapped/human
EMBEDDING_DIM=1280

OUTPUT_BASE=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/results
OUTPUT_FOLDER=${OUTPUT_BASE}/human_esm2
OUTPUT_PREFIX=newtest
FOLDSEEK_FASTA=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/data/r3-ALLSPECIES_foldseekrep_seq.fa
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
        f) FOLDSEEK=1 ;;
        F) FOLDSEEK_FASTA=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done

# Build flags from the path / dim variables
EMBEDDING_FLAG="--embedding ${EMBEDDING}"
EMBEDDING_DIM_FLAG="--input-dim ${EMBEDDING_DIM}"

FOLDSEEK_CMD=""
if [ -n "$FOLDSEEK" ]; then
    FOLDSEEK_CMD="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA} --add_foldseek_after_projection"
fi

if [ ! -d "${OUTPUT_FOLDER}" ]; then
    mkdir -p "${OUTPUT_FOLDER}"
fi

dscript train \
    --train "${TRAIN}" \
    --test "${TEST}" \
    ${EMBEDDING_FLAG} \
    ${EMBEDDING_DIM_FLAG} \
    ${TOPSY_TURVY} \
    --outfile "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_results.log" \
    --save-prefix "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_ep" \
    --device "${DEVICE}" \
    --lr 0.0005 \
    --lambda 0.05 \
    --num-epoch 10 \
    --weight-decay 0 \
    --batch-size 25 \
    --pool-width 9 \
    --kernel-width 7 \
    --dropout-p 0.2 \
    --projection-dim 100 \
    --hidden-dim 50 \
    ${FOLDSEEK_CMD}