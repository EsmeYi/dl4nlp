#!/bin/bash
#SBATCH -A naiss2026-4-769
#SBATCH -J dl4nlp_a4
#SBATCH -o a4/logs/%j.out
#SBATCH -e a4/logs/%j.err
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:00:00

module load Python/3.11.3-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_env/bin/activate
export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/hf_cache
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)

pip install langchain langchain-community langchain-huggingface langchain-core \
    langchain-chroma langchain-text-splitters sentence-transformers chromadb scikit-learn \
    -q --disable-pip-version-check

cd /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_assignments

python a4/A4.py
