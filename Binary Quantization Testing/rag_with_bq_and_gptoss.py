from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import requests

from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import JinaEmbeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate

# === Config ===
JINA_API_KEY = "jina_749fcb059c2f422d8ea05b9a1b95f693V5KvVT-7_eUEcs4C3WsyKX-lJJ_M"
QDRANT_URL = "http://192.168.1.13:6333"
COLLECTION_NAME = "With_BQ"
K = 10
LLM_URL = "http://192.168.1.11:8078/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"

app = FastAPI(title="RAG Retriever API")

print("Initializing retrievers and LLM...")

# === Qdrant client & vectorstores ===
client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v3")

# Dense retriever
dense_vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
    content_payload_key="text"
)
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": K})

# Sparse retriever
sparse_model = FastEmbedSparse(model_name="qdrant/bm25")
sparse_vectorstore = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    sparse_embedding=sparse_model,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse",
    content_payload_key="text"
)
sparse_retriever = sparse_vectorstore.as_retriever(search_kwargs={"k": K})

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# === Prompt Template ===
template = """You are an expert assistant. Use the context below to reason step by step before giving a final answer. Think logically and only answer based on the provided information.

Context:
{context}

Question: {question}

Think step by step, then provide your final answer clearly marked as: "Answer: <final answer>"."""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("Retrievers and LLM ready.")

# === API Schema ===
class RAGGenerateRequest(BaseModel):
    query: str

class RAGGenerateResponse(BaseModel):
    answer: str
    context: List[str]
    retrieval_time: float
    formatting_time: float
    generation_time: float
    total_time: float

# === Endpoint ===
@app.post("/rag/generate", response_model=RAGGenerateResponse)
async def generate_rag(request: RAGGenerateRequest):
    total_start = time.time()

    # Step 1: Retrieve documents
    retrieval_start = time.time()
    docs = ensemble_retriever.invoke(request.query)
    retrieval_time = time.time() - retrieval_start

    # Step 2: Format context
    formatting_start = time.time()
    context_str = format_docs(docs)
    formatting_time = time.time() - formatting_start

    # Step 3: Format prompt
    formatted_prompt = prompt.invoke({
        "context": context_str,
        "question": request.query
    }).to_string()

    # Step 4: Call LLM
    generation_start = time.time()
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": formatted_prompt}
        ]
    }
    response = requests.post(
        LLM_URL,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    if response.status_code != 200:
        raise RuntimeError(f"LLM returned error: {response.status_code} - {response.text}")
    
    data = response.json()
    if "choices" in data and len(data["choices"]) > 0:
        answer = data["choices"][0]["message"]["content"].strip()
    else:
        answer = "No valid response from model."

    generation_time = time.time() - generation_start
    total_time = time.time() - total_start

    return RAGGenerateResponse(
        answer=answer,
        context=[doc.page_content for doc in docs],
        retrieval_time=retrieval_time,
        formatting_time=formatting_time,
        generation_time=generation_time,
        total_time=total_time
    )