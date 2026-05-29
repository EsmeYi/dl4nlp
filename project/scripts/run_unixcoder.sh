#!/bin/bash
#SBATCH -A NAISS2026-4-769
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:30:00
#SBATCH -J unixcoder_baseline
#SBATCH -o /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/unixcoder_%j.log
#SBATCH -e /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/unixcoder_%j.log

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/lirongy/Alvis/dl4nlp_env/bin/activate
cd /cephyr/users/lirongy/Alvis/dl4nlp/Experiment

export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p $TRANSFORMERS_CACHE

python -u unixcoder_baseline.py
