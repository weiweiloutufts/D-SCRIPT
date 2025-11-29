#!/bin/bash
#SBATCH --job-name=dscript              
#SBATCH -p batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.env.%A_%a.out
#SBATCH --error=logs/%j.env.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu

module purge 
module load anaconda/2024.10
conda env create -f ../environment.yml



