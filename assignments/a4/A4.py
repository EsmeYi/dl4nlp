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

# 🎓 Task 3.1: Embedding model
# MiniLM-L6-v2: text -> 384-dim vectors, trained for semantic similarity
# same model for docs + questions -> shared vector space -> semantic search
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

# 🎓 Task 3.3: Build Chroma vector store
# embed all chunks -> store in Chroma with cosine similarity (angle only, ignores magnitude)
# cosine more robust than Euclidean: text vectors from long docs have larger magnitudes
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

# ★ 🎓 Task 4.1: Defining the full RAG pipeline
# baseline Llama 49% (guesses from general knowledge) -> RAG 69%
# (1) MiniLM embeds all abstracts -> Chroma (cosine); chunk 512 chars / 50 overlap (MiniLM max 256 tokens)
# (2) before_model: embed question -> search Chroma k=1 -> augmented prompt "Answer Yes/No based on context..."
# (3) Llama reads augmented prompt only -> outputs Yes/No
# 96/100 retrieval accuracy -> bottleneck is LLM not search; F1 0.756
class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]               # get the user's question
        retrieved_docs = self.vector_store.similarity_search(last_message.content, k=1)
        # embed the question and find the 1 most similar chunk in Chroma (cosine similarity)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  # chunk text

        augmented_message_content = (
            f"Answer the following medical question with only 'Yes' or 'No', "
            f"based on the provided context.\n\n"
            f"Context:\n{docs_content}\n\n"        # retrieved paper chunk goes here
            f"Question: {last_message.content}\n\n"  # original question
            f"Answer (Yes or No):"                  # prompt the model to answer concisely
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            # replace the original bare question with the augmented prompt
            # LLM only ever sees this version — never the bare question
            "context": retrieved_docs,             # save retrieved docs for Task 5.2 evaluation
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

# 🎓 Task 5.1: Evaluate RAG vs no-retrieval baseline
# use_rag=True/False; extract yes/no by response.startswith; compare to gold label
# F1 needed: class imbalance -> always-yes model gets high accuracy but is useless
# RAG: acc=0.69, F1=0.756; baseline: acc=0.49
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

# 🎓 Task 5.2: Check if correct documents were retrieved
# compare retrieved chunk's doc_id vs gold_document_id -> 96/100 correct
# LLM still gets 31% wrong: relevant sentence in different chunk, or misreads medical text
# to improve: k>1 chunks or larger/medical LLM
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
