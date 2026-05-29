# Cross-Modal Contrastive Alignment of C Source Code and x86-64 Assembly

**DL4NLP Course Project — Chalmers University of Technology, 2025**

This project investigates whether a lightweight Transformer can align C source code with its compiled x86-64 assembly in a shared embedding space, supervised by hardware performance labels (IPC from `llvm-mca`). We additionally extend the alignment to natural-language performance descriptions, enabling plain-English hardware-aware code search.

---

## Task Overview

- **Task 1 — Src↔Asm retrieval**: given a C function, retrieve its assembly counterpart (and vice versa) from a pool of 30,991 candidates.
- **Task 2 — NL→Code retrieval**: given a plain-English performance description (e.g. *"memory-bound function with sequential access pattern"*), retrieve the matching C function from a 2,000-candidate pool.
- **Task 3 — IPC prediction**: a linear probe on frozen source embeddings predicts log(1+IPC); reported as R².

**Dataset**: [AnghaBench](https://github.com/brenocfg/AnghaBench) (309,913 (C source, x86-64 asm, IPC) triples after filtering). NL descriptions are generated with Gemini 2.0 Flash.

---

## Repository Structure

```
project/
├── prepare.py              # Data loading, IA tokenizer, evaluation utilities
│
├── train_base.py           # Src↔Asm base model (InfoNCE + IPC regression head)
├── train_improved.py       # + IPC-aware label smoothing (soft InfoNCE)
├── train_nl.py             # Three-way NL+Src+Asm extension (two-stage training)
├── train_scaling.py        # Data scaling curve (--fraction argument)
│
├── baselines/
│   ├── codebert_baseline.py        # CodeBERT fine-tuned / zero-shot
│   ├── unixcoder_baseline.py       # UniXcoder fine-tuned / zero-shot
│   ├── static_feature_baseline.py  # 14 hand-crafted features → Ridge regression
│   └── eval_bm25_baseline.py       # BM25 lexical retrieval
│
├── eval/
│   ├── eval_hw_aware.py        # Hardware-aware retrieval precision (IPC bucket analysis)
│   └── eval_probe_extended.py  # Extended IPC probe (multiple regressors)
│
├── data_gen/
│   ├── generate_triplets.py    # Build (S, A, IPC) dataset from AnghaBench via LLVM
│   └── gen_nl_descriptions.py  # Generate NL descriptions with Gemini 2.0 Flash
│
├── scripts/                    # SLURM batch scripts for the Alvis cluster
│   ├── run_base_seeded.sh      # 3-seed Src↔Asm training (array: 42,43,44)
│   ├── run_improved_seeded.sh  # 3-seed improved model
│   ├── run_scaling.sh          # Scaling curve (array: 5,10,25,50,75,100)
│   ├── run_nl.sh               # NL extension training
│   ├── run_unixcoder.sh        # UniXcoder baseline
│   └── run_static_features.sh  # Static features baseline
│
└── results/
    ├── master_results.json             # Consolidated results for all experiments
    ├── improved_results_seed{42,43,44}.json   # Per-seed improved model results
    ├── scaling_results_{5,10,25,50,75,100}pct_seed42.json
    ├── static_feature_results.json
    └── hw_aware_tsne.pdf               # t-SNE coloured by IPC bucket
```

---

## Method

### Model

Two identical Transformer encoders (4 layers, 4 heads, d=256, proj=128-dim; **10.9M params total**):
- **Source encoder**: BPE tokenizer (vocab 16,384) → mean-pooled → L2-normalized 128-dim embedding.
- **Assembly encoder**: Instruction-Aware (IA) tokenizer (vocab ~215; opcode → unique ID, registers → class tokens `[GPR]`/`[SIMD]`) → same architecture.

### Loss

```
L = L_InfoNCE(src, asm) + λ_hw · MSE(hw_head(src_emb), IPC)
```

`λ_hw = 0.15`, temperature `τ = 0.05`.

**IPC-aware label smoothing** (`train_improved.py`): soft InfoNCE targets replace the hard diagonal with a label-smoothing mixture:
```
target[i,j] = (1 - α) · δ(i=j) + α · softmax(-|log(1+IPC_i) - log(1+IPC_j)| / σ)
```
`α = 0.05`, `σ = 0.5`. Pairs with similar IPC are mildly discounted as negatives.

### NL Extension (`train_nl.py`)

Two-stage training on top of a pretrained Src↔Asm checkpoint:
1. **Stage 1 (NL warm-up)**: freeze Src/Asm encoders; train only NL projection.
2. **Stage 2 (joint fine-tuning)**: unfreeze all; NL projection LR = 5×10⁻⁴, Src/Asm encoders LR = 5×10⁻⁵.

---

## Results Summary

### Src↔Asm Retrieval (test set, n=30,991)

| Model | Params | R@1 | R@5 | R@10 | Probe R² |
|-------|-------:|-----|-----|------|----------|
| Static features (Ridge) | — | — | — | — | 0.063 |
| CodeBERT zero-shot | 128M | — | — | — | 0.237 |
| UniXcoder zero-shot | 129M | — | — | — | 0.160 |
| CodeBERT fine-tuned | 128M | 0.463 | 0.746 | 0.838 | 0.784 |
| UniXcoder fine-tuned | 129M | **0.547** | **0.798** | **0.883** | **0.821** |
| **Ours (base, 3-seed mean)** | **10.9M** | 0.640 | 0.863 | 0.919 | 0.716 |
| **Ours + label smoothing** | **10.9M** | 0.644 | 0.866 | 0.917 | 0.717 |

Our 10.9M-parameter model achieves R@1=0.640 — outperforming fine-tuned CodeBERT (128M, R@1=0.463) with 12× fewer parameters. The linear probe reaches R²=0.716 (11.4× over the 14-feature static baseline), demonstrating that micro-architectural intuition transfers across modalities through contrastive alignment alone.

### Data Scaling (seed=42, full model)

| Training data | Samples | R@1 | R@10 | Probe R² |
|:---:|---:|-----|------|----------|
| 5% | 15,495 | 0.334 | 0.714 | 0.524 |
| 10% | 30,991 | 0.445 | 0.808 | 0.611 |
| 25% | 77,478 | 0.572 | 0.885 | 0.655 |
| 50% | 154,956 | 0.629 | 0.916 | 0.689 |
| 75% | 232,434 | 0.632 | 0.916 | 0.711 |
| 100% | 309,913 | 0.656 | 0.923 | 0.719 |

Performance improves steeply up to 50% of the corpus, then plateaus — suggesting further gains from larger datasets.

### NL→Code Retrieval (Task 1, 2,000-candidate pool)

| Model | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| BM25 | 0.062 | 0.131 | 0.171 |
| CodeBERT zero-shot | ~0.002 | — | — |
| **Ours (3-way NL+Src+Asm)** | **0.352** | — | — |

Full consolidated results: [`results/master_results.json`](results/master_results.json)

---

## Reproducing Experiments

> Scripts are written for the **Alvis HPC cluster** (SLURM + A40 GPU). Paths are hardcoded to `/cephyr/users/lirongy/Alvis/`. Adapt `HF_HOME`, module loads, and account strings before running elsewhere.

### Environment

```bash
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
pip install transformers rank_bm25
```

The dataset (`prepare.py`) expects AnghaBench-derived HDF5/JSON files at a path set by `DATA_ROOT` in `prepare.py`.

### Training

```bash
# Base model (3 seeds)
sbatch scripts/run_base_seeded.sh

# Improved model with IPC-aware label smoothing (3 seeds)
sbatch scripts/run_improved_seeded.sh

# Data scaling curve (6 points: 5/10/25/50/75/100%)
sbatch scripts/run_scaling.sh

# NL extension (requires a base checkpoint)
sbatch scripts/run_nl.sh
```

### Baselines

```bash
sbatch scripts/run_unixcoder.sh        # UniXcoder fine-tuned
sbatch scripts/run_static_features.sh  # 14-feature Ridge baseline
python baselines/eval_bm25_baseline.py # BM25 (runs on CPU, no GPU needed)
```

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| IA tokenizer (vocab ~215) | Preserves opcode atomicity; BPE splits `vaddps` into sub-word tokens that lose hardware semantics |
| Shared encoder architecture | Parameter efficiency; assembly and source share the same latent geometry |
| IPC regression auxiliary task (λ=0.15) | Encourages embeddings to encode micro-architectural information beyond syntactic similarity |
| Label smoothing on IPC similarity | Prevents over-penalising functionally similar but textually different pairs |
| Two-stage NL training | Warm-up avoids collapsing the pretrained Src↔Asm space before NL is grounded |
