# Assignment 4 — Retrieval-Augmented Generation (RAG)

RAG pipeline using LangChain (Option A: RAG Agent) to answer medical yes/no questions from the PubMedQA dataset.

## Setup

- **LM**: meta-llama/Llama-3.1-8B-Instruct
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector store**: Chroma (cosine similarity)
- **Chunk size**: 512 tokens, 50 token overlap
- **Dataset**: PubMedQA — medical questions with yes/no gold labels

## Pipeline

```
Question → MiniLM embed → Chroma retrieval (k=1)
         → augmented prompt (context + question)
         → Llama 3.1-8B → Yes/No answer
```

## Results (100 questions)

| | RAG | No retrieval baseline |
|---|---|---|
| Accuracy | **0.690** | 0.490 |
| F1 | **0.756** | 0.523 |
| Retrieval accuracy | **96/100 (96%)** | — |

RAG improves accuracy from 0.49 (near random) to 0.69 by grounding answers in retrieved paper abstracts. The vector store retrieves the correct document in 96% of cases.

## Files

- `A4.py` — full RAG pipeline implementation
- `run.sh` — Slurm job script (A40 GPU, 2h limit)
- `ori_pqal.json` — PubMedQA dataset
- `logs/` — job output logs
