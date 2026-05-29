#!/bin/bash
#SBATCH -A NAISS2026-4-769
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:30:00
#SBATCH -J scaling
#SBATCH -o /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/scaling_%a_%j.log
#SBATCH -e /cephyr/users/lirongy/Alvis/dl4nlp/Experiment/scaling_%a_%j.log
#SBATCH --array=5,10,25,50,75,100

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/lirongy/Alvis/dl4nlp_env/bin/activate
cd /cephyr/users/lirongy/Alvis/dl4nlp/Experiment

export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p $TRANSFORMERS_CACHE

# SLURM_ARRAY_TASK_ID is 25/50/75/100 → fraction = task_id / 100
FRACTION=$(python3 -c "print(${SLURM_ARRAY_TASK_ID}/100)")
python -u final_train_scaling.py --fraction $FRACTION --seed 42
