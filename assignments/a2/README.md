# Assignment 2 — Transformer Language Model

Implementation of a Transformer-based language model following the OLMo-2 architecture, trained on the same Wikipedia dataset as Assignment 1.

## Architecture

- **Model**: Decoder-only Transformer (OLMo-2 style)
- **Components**: RMSNorm (Pre-Norm), SwiGLU MLP, Multi-Head Attention with RoPE, causal masking
- **Config**: hidden_size=256, intermediate_size=512, 8 attention heads, 4 layers, max_pos=256
- **Parameters**: ~6M
- **Tokenizer**: Reused from A1 (word-level, vocab built from training data)

## Training Results

| Epoch | Val Perplexity |
|-------|---------------|
| 1     | ~80           |
| 2     | ~60           |
| 3     | **51.1**      |

Trained for 3 epochs with AdamW (lr=1e-3), batch size 64, on Alvis cluster (T4 GPU).

Compared to A1 RNN (val_ppl=66.4), the Transformer achieves lower perplexity with a similar parameter count, demonstrating the effectiveness of attention over recurrence.

## OLMo-2 Comparison (Task 3.3)

Generated with temperature=0.7, top-k=50.

---

**Prompt:** `In natural language processing, a Transformer`

**Our model:**
```
in natural language processing , a <UNK> <EOS> ( <UNK> ) is a <UNK> to a <UNK> , a <UNK> . the <UNK> is a <UNK> <UNK> with a <UNK> .
```

**OLMo-2 (1B):**
```
is a type of neural network that enables a computer to perform various tasks. It is a machine
learning model that enables it to learn from data and predict outcomes. In machine learning, a
transformer learns to make predictions by processing data in a linear sequence.
```

---

**Prompt:** `Stockholm is the capital of Sweden . The country is known for`

**Our model:**
```
stockholm is the capital of sweden . the country is known for <EOS> <UNK> , of which the capital of cologne .
```

**OLMo-2 (1B):**
```
its unique design and for a wealth of cultural heritage. Sweden is known as the land of the
midnight sun with large areas of northern Swedish territory illuminated for several months of
the year. Stockholm, the capital city, has a population of 1.5 million people.
```

---

**Prompt:** `The most important invention of the 20th century was`

**Our model:**
```
the most important invention of the 20th century was <EOS> <UNK> , which was built in <UNK> <UNK>
and <UNK> in <UNK> , england . the <UNK> <UNK> was <UNK> of <UNK> ...
```

**OLMo-2 (1B):**
```
the telephone. Although it was invented in the 19th century, it wasn't as popular as it is
today. The telephone is the most important invention of this century because it has made
communication easier.
```

---

### Analysis

Our model's output is dominated by `<UNK>` tokens because our word-level vocabulary only covers frequent words from the Wikipedia training set — rare and domain-specific words are unknown. OLMo-2 uses a BPE subword tokenizer that can represent any word and never produces `<UNK>`.

Despite this, our model has learned some syntactic patterns (comma placement, relative clauses, determiners), which is reflected in its val_ppl of 51.1. The gap in output quality reflects three key differences:

| | Our Transformer | OLMo-2 |
|---|---|---|
| Parameters | ~6M | ~1B |
| Training tokens | ~150k paragraphs | Hundreds of billions |
| Tokenization | Word-level (UNK) | BPE subword |

## Files

- `A2_skeleton.py` — model implementation (RMSNorm, SwiGLU, MHA+RoPE, Transformer)
- `train.sh` — Slurm training script
- `compare_olmo.sh` — Slurm comparison script vs OLMo-2
- `trainer_output/` — saved model checkpoint
