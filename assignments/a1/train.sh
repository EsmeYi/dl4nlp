#!/bin/bash
#SBATCH -A naiss2026-4-769
#SBATCH -J dl4nlp_a1
#SBATCH -o a1/logs/%j.out
#SBATCH -e a1/logs/%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 01:00:00

module load Python/3.11.3-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_env/bin/activate
export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/hf_cache

cd /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_assignments

python - <<'EOF'
import sys, nltk
sys.path.insert(0, 'a1')
nltk.download('punkt_tab', quiet=True)

import torch
from transformers import TrainingArguments
from A1_skeleton import build_tokenizer, load_datasets, A1RNNModelConfig, A1RNNModel, A1Trainer

tokenizer = build_tokenizer('a1/train.txt', max_voc_size=10000, model_max_length=256)
tokenizer.save('a1/tokenizer.pkl')
print(f'Vocab size: {len(tokenizer)}')

dataset = load_datasets('a1/train.txt', 'a1/val.txt')
print(f'Train: {len(dataset["train"])}  Val: {len(dataset["val"])}')

config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=256, hidden_size=512)
model = A1RNNModel(config)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

args = TrainingArguments(
    output_dir='a1/trainer_output',
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
