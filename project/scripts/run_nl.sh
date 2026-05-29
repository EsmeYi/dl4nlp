#!/bin/bash
#SBATCH -A NAISS2026-4-769
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:30:00
#SBATCH -J final_nl_v3
#SBATCH -o /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/final_nl_v3_%j.log
#SBATCH -e /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/final_nl_v3_%j.log

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/lirongy/Alvis/dl4nlp_env/bin/activate
cd /cephyr/users/lirongy/Alvis/dl4nlp/Experiment

export HF_HOME=/cephyr/users/lirongy/Alvis/.cache/huggingface
export TRANSFORMERS_CACHE=/cephyr/users/lirongy/Alvis/.cache/huggingface

python -u final_train_nl_v3.py
