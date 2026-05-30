#!/bin/bash
#SBATCH -A naiss2026-4-769
#SBATCH -J dl4nlp_a3
#SBATCH -o a3/logs/%j.out
#SBATCH -e a3/logs/%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 02:00:00

module load Python/3.11.3-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_env/bin/activate
export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/hf_cache

pip install evaluate rouge_score -q --disable-pip-version-check

cd /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_assignments

python a3/A3_skeleton.py
