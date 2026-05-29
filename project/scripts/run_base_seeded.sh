#!/bin/bash
#SBATCH -A NAISS2026-4-769
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:00:00
#SBATCH --array=42,43,44
#SBATCH -J multiseed_%a
#SBATCH -o /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/multiseed_%a.log
#SBATCH -e /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/multiseed_%a.log

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/lirongy/Alvis/dl4nlp_env/bin/activate
cd /cephyr/users/lirongy/Alvis/dl4nlp/Experiment

python -u final_train_seeded.py --seed ${SLURM_ARRAY_TASK_ID}
