# DL4NLP Assignments — Lirong Yi

Four assignments covering language model architectures, fine-tuning, and retrieval-augmented generation.

| Assignment | Topic | Key Result |
|---|---|---|
| [A1](a1/) | RNN Language Model | val perplexity = 66.4 |
| [A2](a2/) | Transformer Language Model | val perplexity = 51.1 |
| [A3](a3/) | Supervised Fine-Tuning + LoRA | ROUGE-L = 0.674 (SFT), 0.629 (LoRA) |
| [A4](a4/) | Retrieval-Augmented Generation | accuracy 0.49 → 0.69 with RAG |

---

## AI Tool Usage

I used Claude (claude.ai) as an assistant throughout the assignments — primarily to clarify concepts and debug issues, and occasionally to help write boilerplate code.

- **A1**: Used AI to understand teacher forcing and the loss shift logic; implemented the tokenizer and training loop myself with that understanding
- **A2**: Used AI to clarify Q/K/V roles and the √d_h scaling; also used it to debug dependency conflicts between PyTorch 2.1 and the transformers version required for OLMo-2
- **A3**: Used AI to understand ChatML format and the LoRA initialization (why B=0, why A uses Kaiming); wrote the LoRA layer and data formatting code with that understanding
- **A4**: Used AI to understand the chunking strategy and LangChain's middleware pattern; used it to fix a LangChain API change (`langchain.text_splitter` → `langchain_text_splitters`)
