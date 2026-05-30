#!/bin/bash
#SBATCH -A naiss2026-4-769
#SBATCH -J dl4nlp_a2
#SBATCH -o a2/logs/%j.out
#SBATCH -e a2/logs/%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 01:30:00

module load Python/3.11.3-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_env/bin/activate
export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/hf_cache

cd /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_assignments

python - <<'EOF'
import sys, nltk
sys.path.insert(0, 'a1')
sys.path.insert(0, 'a2')
nltk.download('punkt_tab', quiet=True)

from transformers import TrainingArguments
from A1_skeleton import A1Tokenizer, load_datasets, A1Trainer
from A2_skeleton import A2ModelConfig, A2Transformer

# Reuse tokenizer from A1
tokenizer = A1Tokenizer.from_file('a1/tokenizer.pkl')
dataset   = load_datasets('a1/train.txt', 'a1/val.txt')
print(f'Train: {len(dataset["train"])}  Val: {len(dataset["val"])}')

config = A2ModelConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    intermediate_size=512,
    num_attention_heads=8,
    num_hidden_layers=4,
    rms_norm_eps=1e-6,
    rope_theta=10000,
    max_position_embeddings=256,
    hidden_act='silu',
)
model = A2Transformer(config)
import torch
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')

args = TrainingArguments(
    output_dir='a2/trainer_output',
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-3,
    optim='adamw_torch',
    eval_strategy='epoch',
)

trainer = A1Trainer(model=model, args=args,
                    train_dataset=dataset['train'],
                    eval_dataset=dataset['val'],
                    tokenizer=tokenizer)
trainer.train()
EOF
