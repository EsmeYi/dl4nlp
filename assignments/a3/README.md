# Assignment 3 — Supervised Fine-Tuning with LoRA

Fine-tuning SmolLM2-135M on the SmolTalk instruction dataset using SFT and LoRA.

## Task

Turn a pretrained language model (pure text continuation) into an instruction-following assistant by training on instruction-response pairs from the [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) dataset.

## Setup

- **Base model**: HuggingFaceTB/SmolLM2-135M
- **Dataset**: SmolTalk (`all` subset), 5000 train / 400 test samples
- **Format**: ChatML (`<|im_start|>role\ncontent<|im_end|>`)
- **Loss**: Cross-entropy on response tokens only (prompt masked with -100)

## Results

| Model | eval_loss | ROUGE-L | Trainable params |
|-------|-----------|---------|-----------------|
| Pretrained (no training) | 2.619 | 0.575 | 0 |
| Full SFT (1 epoch) | **1.157** | **0.674** | 134,515,008 (135M) |
| LoRA (r=8, α=16, 1 epoch) | 1.580 | 0.629 | **921,600 (0.9M)** |

Trained on Alvis cluster (T4 GPU). Full SFT: ~13 min, LoRA: ~11 min.

## LoRA Configuration

- **Rank r = 8**, scaling **α = 16**
- Target layers: `q_proj`, `k_proj`, `v_proj`, `o_proj` in all attention blocks
- All other parameters frozen
- LoRA uses **0.7%** of full SFT parameters while achieving 90% of the ROUGE-L gain

## Qualitative Comparison

**Prompt:** `Correct the verb tense: "I swim every day last week."`

- **Pretrained**: continues the text unrelated to the instruction
- **Full SFT**: `"I swam every day last week."` ✓ (but sometimes repeats)
- **LoRA**: `"I swam every day last week."` ✓ (but appends garbled tokens due to limited capacity)

## Files

- `A3_skeleton.py` — full implementation (data formatting, tokenization, SFT, LoRA)
- `train.sh` — Slurm job script (T4 GPU, 2h limit)
- `logs/` — training output and error logs
