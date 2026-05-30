"""
Assignment 4: Retrieval-Augmented Generation (RAG)
Option A: RAG Agent using LangChain
"""

import json
import pandas as pd
from typing import Any

# ============================================================
# Part 1: Load the dataset
# ============================================================

tmp_data = pd.read_json("a4/ori_pqal.json").T
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({
    "abstract": tmp_data.apply(lambda row: " ".join(row.CONTEXTS + [row.LONG_ANSWER]), axis=1),
    "year": tmp_data.YEAR
})

questions = pd.DataFrame({
    "question":        tmp_data.QUESTION,
    "year":            tmp_data.YEAR,
    "gold_label":      tmp_data.final_decision,
    "gold_context":    tmp_data.LONG_ANSWER,
    "gold_document_id": documents.index
})

print(f"Documents: {len(documents)}, Questions: {len(questions)}")
print("\nExample question:")
print(questions.iloc[0].question)
print("\nExample document (first 300 chars):")
print(documents.iloc[0].abstract[:300])

# ============================================================
# Part 2: Load the language model
# ============================================================

from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print(f"\nLoading LM: {MODEL_NAME}")
llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "return_full_text": False,
        "temperature": 0.1,
        "do_sample": True,
    },
    model_kwargs={"torch_dtype": "float16"},
    device=0,  # GPU
)

# Sanity check
print("\nSanity check — LM output:")
print(llm.invoke("What is the capital of France? Answer in one word."))

# ============================================================
# Part 3: Build the vector store
# ============================================================

# Task 3.1: Embedding model
from langchain_huggingface import HuggingFaceEmbeddings

print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

# Sanity check
test_vec = embedding_model.embed_query("What is programmed cell death?")
print(f"Embedding shape: {len(test_vec)}")

# Task 3.2: Chunk the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\nChunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(
    texts=documents.abstract.tolist(),
    metadatas=metadatas,
)
chunks = text_splitter.split_documents(texts)
print(f"Total chunks: {len(chunks)}")
print(f"Example chunk:\n{chunks[0].page_content[:200]}")

# Task 3.3: Build Chroma vector store
from langchain_chroma import Chroma

print("\nBuilding vector store (this may take a few minutes)...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Sanity check
results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
print("\nVector store sanity check:")
for res, score in results:
    print(f"  [SIM={score:.3f}] {res.page_content[:100]} [{res.metadata}]")

# ============================================================
# Part 4: RAG Agent (Option A)
# ============================================================

from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage

class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = self.vector_store.similarity_search(last_message.content, k=1)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"Answer the following medical question with only 'Yes' or 'No', "
            f"based on the provided context.\n\n"
            f"Context:\n{docs_content}\n\n"
            f"Question: {last_message.content}\n\n"
            f"Answer (Yes or No):"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    middleware=[RetrieveDocumentsMiddleware(vector_store)],
)

# Sanity check
test_question = questions.iloc[0].question
print(f"\nSanity check — RAG Agent:")
print(f"Question: {test_question}")
for step in agent.stream(
    {"messages": [{"role": "user", "content": test_question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# ============================================================
# Part 5: Evaluation
# ============================================================

# Task 5.1: Evaluate RAG vs no-retrieval baseline
def extract_yes_no(text):
    """Extract Yes/No from model output."""
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    return None


def evaluate_rag(questions_sample, use_rag=True):
    """Run evaluation on a sample of questions."""
    preds, labels, valid = [], [], []
    retrieved_ids = []

    for i, (idx, row) in enumerate(questions_sample.iterrows()):
        if i % 50 == 0:
            print(f"  {i}/{len(questions_sample)}")

        if use_rag:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": row.question}]}
            )
            answer_text = result["messages"][-1].content
            # Get retrieved doc id from context if available
            context = result.get("context", [])
            if context:
                retrieved_ids.append(context[0].metadata.get("id", None))
            else:
                retrieved_ids.append(None)
        else:
            prompt = (
                f"Answer the following medical question with only 'Yes' or 'No'.\n"
                f"Question: {row.question}\nAnswer (Yes or No):"
            )
            answer_text = llm.invoke(prompt)
            retrieved_ids.append(None)

        pred = extract_yes_no(answer_text)
        if pred is not None:
            preds.append(pred)
            labels.append(row.gold_label)
            valid.append(idx)

    # Compute accuracy and F1
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, pos_label="yes")
    print(f"  Valid answers: {len(preds)}/{len(questions_sample)}")
    print(f"  Accuracy: {acc:.3f}  F1: {f1:.3f}")
    return preds, labels, valid, retrieved_ids


# Use a subset for speed (full evaluation can take a long time)
EVAL_SAMPLE = 100
sample = questions.iloc[:EVAL_SAMPLE]

print(f"\n{'='*60}")
print(f"EVALUATING RAG (n={EVAL_SAMPLE})")
print('='*60)
rag_preds, rag_labels, rag_valid, rag_retrieved = evaluate_rag(sample, use_rag=True)

print(f"\n{'='*60}")
print(f"EVALUATING BASELINE (no retrieval, n={EVAL_SAMPLE})")
print('='*60)
base_preds, base_labels, base_valid, _ = evaluate_rag(sample, use_rag=False)

# Task 5.2: Check if correct documents were retrieved
correct_retrieval = 0
total_retrieval = 0
for idx, ret_id in zip(rag_valid, rag_retrieved):
    if ret_id is not None:
        total_retrieval += 1
        gold_id = questions.loc[idx, "gold_document_id"]
        if ret_id == gold_id:
            correct_retrieval += 1

if total_retrieval > 0:
    print(f"\nRetrieval accuracy (correct doc fetched): "
          f"{correct_retrieval}/{total_retrieval} = {correct_retrieval/total_retrieval:.3f}")

# Inspect a few examples
print(f"\n{'='*60}")
print("SAMPLE OUTPUTS")
print('='*60)
for i in range(min(3, len(rag_valid))):
    idx = rag_valid[i]
    row = questions.loc[idx]
    print(f"\nQuestion: {row.question}")
    print(f"Gold:     {row.gold_label}")
    print(f"RAG pred: {rag_preds[i]}")
    print(f"Retrieved doc id: {rag_retrieved[i]}")
    print(f"Gold doc id:      {row.gold_document_id}")
