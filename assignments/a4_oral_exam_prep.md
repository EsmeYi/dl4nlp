# Assignment 4 — Oral Exam Preparation

## Overview

**Q: What is RAG and why do we need it?**

> RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model. Instead of relying only on the model's internal knowledge (which can be outdated or incomplete), RAG first retrieves relevant documents from a database and provides them as context when generating an answer. This reduces hallucination and allows the model to answer questions about specialized domains (like medicine) more accurately.

**Q: What problem does RAG solve that standard LLMs cannot?**

> Standard LLMs have a fixed knowledge cutoff and cannot access external information at inference time. They also tend to hallucinate — generating plausible-sounding but incorrect facts. RAG grounds the model's answer in retrieved evidence, making it more accurate and verifiable.

---

## Part 1: Dataset

**Q: What is the PubMedQA dataset?**

> PubMedQA is a medical question-answering dataset built from PubMed research papers. Each example contains a question derived from a paper's title, the paper's abstract as context, and a yes/no answer based on the paper's conclusion. We use it to evaluate whether RAG helps a model answer medical yes/no questions correctly.

**Q: What are `documents` and `questions` in this assignment?**

> - `documents`: a table of paper abstracts (CONTEXTS + LONG_ANSWER concatenated). This is the retrieval corpus — the database RAG searches.
> - `questions`: a table of medical questions with gold yes/no labels and the ID of the document that contains the answer.

---

## Part 2: Language Model

**Q: How do we load a HuggingFace model in LangChain?**

> Using `HuggingFacePipeline.from_model_id`, which wraps a HuggingFace `pipeline` object inside a LangChain-compatible interface. We set `return_full_text=False` so the model only returns the newly generated text, not the full prompt.

---

## Part 3: Vector Store

**Q: What is an embedding model and why do we need one?**

> An embedding model converts text into a fixed-size vector of numbers. Texts with similar meaning end up with similar vectors (measured by cosine similarity). We need it to enable semantic search: instead of matching keywords exactly, we can find documents that are conceptually related to the query, even if they use different words.

**Q: What is chunking and why is it necessary?**

> Chunking splits long documents into smaller pieces. Embedding models have a maximum input length (typically 256–512 tokens). Documents that are too long must be split so they fit. Smaller chunks also give more precise retrieval — returning exactly the relevant paragraph rather than an entire paper.

**Q: How does the chunk size affect RAG quality?**

> - **Too large**: chunks exceed the embedding model's context window and may contain irrelevant content mixed with relevant content, making retrieval less precise.
> - **Too small**: chunks may lack enough context for the model to answer the question.
> - A good chunk size balances coverage and precision. Overlap between chunks (e.g. 50 tokens) helps avoid splitting relevant sentences across chunk boundaries.

**Q: What is a vector store (Chroma) and how does retrieval work?**

> A vector store stores documents alongside their embeddings. At query time:
> 1. The query is embedded into a vector.
> 2. The store finds the documents whose vectors are most similar (by cosine similarity).
> 3. The top-k most similar documents are returned.
> Chroma is an in-memory vector database that makes this fast and easy to set up.

---

## Part 4: RAG Agent

**Q: What does the RAG Agent do differently from a plain LLM call?**

> A plain LLM call passes the question directly to the model. The RAG Agent intercepts the query, searches the vector store for relevant documents, augments the prompt with those documents as context, and only then passes the enriched prompt to the LLM. This gives the model evidence to base its answer on.

**Q: What does the `RetrieveDocumentsMiddleware` do?**

> It runs before the model receives the message. It:
> 1. Takes the user's question from the message.
> 2. Searches the vector store for the most relevant document chunk.
> 3. Builds a new prompt: `"Based on the following context, answer Yes or No: {context}\n\nQuestion: {question}"`
> 4. Replaces the original message with this augmented prompt.
> The model then only sees the augmented version.

**Q: Why do we prompt the model to answer only "Yes" or "No"?**

> For evaluation in Part 5, we need to parse the model's output into a binary label. If we don't constrain the output, the model may generate long explanations where the yes/no answer is buried or absent. By explicitly asking for "Yes" or "No" at the start, we make evaluation reliable.

---

## Part 5: Evaluation

**Q: How do we evaluate the RAG pipeline?**

> We run the pipeline on a set of questions with known gold labels (yes/no). We extract the model's yes/no prediction from the output and compare it to the gold label. We compute:
> - **Accuracy**: fraction of correct predictions
> - **F1 score**: harmonic mean of precision and recall for the "yes" class — better than accuracy when classes are imbalanced

**Q: What is the baseline comparison and why is it important?**

> We run the same LM without any retrieved context (just the question, no documents). Comparing RAG vs no-retrieval shows whether the retrieval actually helps. If RAG has higher F1/accuracy, the retrieved documents are genuinely useful. If not, the model can already answer well from its own knowledge.

**Q: How do we evaluate retrieval quality (Task 5.2)?**

> Each question has a `gold_document_id` — the ID of the paper that contains the answer. After RAG retrieves a document, we check if its ID matches the gold document ID. This measures whether the vector store is finding the right paper. Low retrieval accuracy means the search is failing, which would explain poor QA performance.

**Q: What factors can cause low retrieval accuracy?**

> 1. **Chunk size too large/small**: poor granularity reduces relevance
> 2. **Embedding model mismatch**: the embedding model may not understand medical terminology well
> 3. **Query-document mismatch**: the question may be phrased very differently from the document
> 4. **k too small**: using k=1 means if the top result is wrong, there's no fallback

---

## Key Concepts Summary

| Concept | What it does |
|---------|-------------|
| Embedding model | Converts text → vector for semantic search |
| Chunking | Splits documents to fit embedding model context |
| Chroma | Vector database storing doc embeddings |
| Retrieval | Finds top-k most similar document chunks |
| RAG Agent | Intercepts query, retrieves docs, augments prompt |
| Evaluation | F1/accuracy on yes/no predictions vs gold labels |

---

### Actual results (Slurm job 6703650, Llama 3.1-8B-Instruct, 100 questions)

| | RAG | No retrieval (baseline) |
|---|---|---|
| Accuracy | **0.690** | 0.490 |
| F1 | **0.756** | 0.523 |
| Retrieval accuracy | **96/100 (96%)** | — |

**Q: What do the results show about RAG?**

> RAG significantly outperforms the no-retrieval baseline: accuracy improves from 0.49 to 0.69 and F1 from 0.52 to 0.76. The no-retrieval baseline is close to random guessing (50%), showing that Llama 3.1-8B has limited medical domain knowledge on its own. With retrieved context, it can read the relevant paper and answer correctly.

**Q: What does the 96% retrieval accuracy mean?**

> In 96 out of 100 cases, the vector store returned the exact paper that contains the gold answer. This shows that MiniLM-L6-v2 embeddings + cosine similarity in Chroma is highly effective for this dataset. The 4 failures are cases where the question phrasing is too different from the document text for semantic search to succeed.

**Q: Why does RAG still get 31% of answers wrong despite 96% retrieval accuracy?**

> Even when the correct document is retrieved, the model may still fail to extract the right answer. Reasons include: (1) the relevant sentence is in a chunk that wasn't retrieved, (2) the model misinterprets the context, or (3) the prompt format causes the model to generate an explanation instead of a clean Yes/No.
