#!/bin/bash
#SBATCH -A NAISS2026-4-769
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 00:30:00
#SBATCH -J static_features
#SBATCH --cpus-per-task=4
#SBATCH -o /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/static_features_%j.log
#SBATCH -e /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/static_features_%j.log

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/lirongy/Alvis/dl4nlp_env/bin/activate
cd /cephyr/users/lirongy/Alvis/dl4nlp/Experiment

python -u static_feature_baseline.py
