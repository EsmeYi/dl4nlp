# Assignment 1: RNN Language Model

## Setup

- Vocabulary size: 10,000
- Embedding size: 256, Hidden size: 512
- Model parameters: ~9.3M
- Training data: 147,059 paragraphs | Validation: 17,874 paragraphs

## Training Results (3 epochs, GPU T4)

| Epoch | Train Loss | Val Loss | Val Perplexity |
|-------|-----------|----------|----------------|
| 1     | 4.963     | 4.533    | 93.0           |
| 2     | 4.402     | 4.297    | 73.5           |
| 3     | 4.189     | 4.196    | **66.4**       |

Final perplexity **66.4** (well below the expected 200–300 range).

## Next Word Prediction (Task 5.1)

```
Input: "she lives in san"
→ francisco (10.06), antonio (8.17), diego (7.82), juan (6.83)
```

## Word Embedding Neighbors (Task 5.3)

**"three"** → four, nine, six, twelve, eight ✅ numbers cluster together

**"king"** → sultan, president, iii ✅ roles and titles nearby

**"sweden"** → weak neighbors (cosine ~0.28) — proper nouns need more training data to learn strong geometric relationships

## Files

| File | Description |
|------|-------------|
| `A1_skeleton.py` | Full implementation |
| `tokenizer.pkl` | Saved tokenizer |
| `trainer_output/` | Saved model weights |
| `train.sh` | Slurm job script |
