#!/bin/bash
#SBATCH --job-name=test
#SBATCH -p cellbio-dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_test_%a.out
#SBATCH --error=/hpc/home/wl324/D-SCRIPT/logs/%A_bernett_test_%a.err
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

TOPSY_TURVY=
EMBEDDING_DIR=/hpc/home/wl324/projects/tt3d/data_archive/esm2_bernett
SEQ_DIR=/hpc/home/wl324/projects/tt3d/data_archive/
FOLDSEEK_FASTA=/hpc/home/wl324/tt3d/p_tt3d_new/fasta/full_fasta.fasta
FOLDSEEK_VOCAB=/hpc/home/wl324/D-SCRIPT/data/foldseek_vocab.json
MODEL_PARAMS=""
DEVICE=0
# Set default model here:
MODEL=/hpc/home/wl324/projects/tt3d/data/results/bernett_esm2_train_lr0.0005_lam0.999_wd0.0001/bernett_best_state_dict.pt

usage() {
    echo "USAGE: ./test.sh [-d DEVICE] [-m MODEL] [-T MODEL_TYPE]

    OPTIONS:

    -d DEVICE: Device used
    -m MODEL: The model sav file
    -T: Set this flag if if the model passed by the '-m MODEL' command is a TT3D Model. Unset this for Topsy-Turvy/D-SCRIPT
    "
}


while getopts "d:m:T" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        m) MODEL=${OPTARG}
        ;;
        T) MODEL_PARAMS="${MODEL_PARAMS} --add_foldseek_after_projection --foldseek_vocab ${FOLDSEEK_VOCAB} --foldseek_fasta ${FOLDSEEK_FASTA} --allow_foldseek"
        ;;
        *) usage
        exit 1;
    esac
done

# Construct the folder
OUTPUT_FLD=${MODEL%/*}
OUTPUT_FILE=${MODEL##*/}
OUTPUT_FILE_PREF=${OUTPUT_FILE%.*}
OUTPUT_FOLDER=${OUTPUT_FLD}/eval-${OUTPUT_FILE_PREF}

echo "Output folder: ${OUTPUT_FOLDER}, model: ${MODEL}, DEVICE: ${DEVICE}"
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir $OUTPUT_FOLDER; fi

lam=0.999
lr=0.0005
wd=0.0001

export WANDB_NAME="bernett_test_lr${lr}_lam${lam}_wd${wd}"
export WANDB_TAGS="bernett,aug,tauri,tok,softmax"
export WANDB_RUN_GROUP="tt3d_backbone"
export WANDB_JOB_TYPE="test"

EMBEDDING=${EMBEDDING_DIR}
TEST=${SEQ_DIR}/bernett_test.tsv
OP_FOLDER_ORG=${OUTPUT_FOLDER}/
if [ ! -d ${OP_FOLDER_ORG} ]; then mkdir -p ${OP_FOLDER_ORG}; fi
OP_FILE=${OP_FOLDER_ORG}/${OUTPUT_FILE}
$PY -u -m dscript.commands.evaluate \
--model ${MODEL} \
--embeddings ${EMBEDDING} \
--test ${TEST} \
-d $DEVICE \
-o $OP_FILE \
--log_wandb \
--wandb-entity bergerlab-mit \
--wandb-project tt3d_backbone 

