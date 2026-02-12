import os
import time
import uuid
import logging
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from collections import defaultdict, deque
from dotenv import load_dotenv

import faiss
import numpy as np
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY", "dev-secret-key")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))  # requests per minute

# ----------------------
# Logging configuration
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------
# FastAPI initialization
# ----------------------

app = FastAPI(
    title="RAG API",
    description="Production-ready REST API for Retrieval-Augmented Generation",
    version="1.0.0",
)

# ----------------------
# CORS configuration
# ----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Security
# ----------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


# ----------------------
# Rate limiting
# ----------------------

rate_limit_store = defaultdict(deque)


def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()
    window = 60  # seconds
    timestamps = rate_limit_store[ip]

    while timestamps and timestamps[0] < now - window:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    timestamps.append(now)


# ----------------------
# Models
# ----------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What is the refund policy?")


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    text: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: List[Citation]
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str


class DocumentCreateRequest(BaseModel):
    document_id: str
    text: str


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class ErrorResponse(BaseModel):
    request_id: str
    error: str


@dataclass
class Document:
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    id: int
    text: str
    document_id: str
    metadata: Dict[str, str]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Chunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    id=-1,  # Will be assigned later
                    text=chunk_text,
                    document_id=document.source,
                    metadata=document.metadata.copy(),
                )
            )
            start += self.chunk_size - self.overlap
        return chunks


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            payload = {"model": self.model_name, "input": text}
            response = requests.post("http://localhost:11434/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
            emb = np.array(data["embeddings"][0])
            emb = normalize(emb)
            embeddings.append(emb)
        return np.array(embeddings)


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]):
        embeddings = np.vstack([chunk.embedding for chunk in chunks])
        self.index.add(embeddings)
        start_id = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk.id = start_id + i
            self.chunks.append(chunk)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        return list(zip(indices[0], scores[0]))

    def remove_by_document_id(self, document_id: str):
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if self.chunks:
            embeddings = np.vstack([c.embedding for c in self.chunks])
            self.index.add(embeddings)
            for i, c in enumerate(self.chunks):
                c.id = i


class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "nomic-embed-text",
        llm_model_name: str = "llama3.2",
        chunk_size: int = 500,
        overlap: int = 100,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.llm_model_name = llm_model_name
        self.chunker = Chunker(chunk_size, overlap)
        self.vector_store: Optional[VectorStore] = None
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.document_ids = set()

    def add_document(self, doc_id: str, text: str):
        if doc_id in self.document_ids:
            raise ValueError("Document already exists")
        self.document_ids.add(doc_id)
        document = Document(text=text, source=doc_id, metadata={})
        chunks = self.chunker.chunk_document(document)
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        if self.vector_store is None:
            self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(chunks)

    def delete_document(self, doc_id: str):
        if doc_id not in self.document_ids:
            return False
        self.document_ids.discard(doc_id)
        if self.vector_store:
            self.vector_store.remove_by_document_id(doc_id)
        return True

    def query(self, question: str):
        if not self.vector_store:
            return "No documents ingested yet.", []
        query_embedding = self.embedding_model.embed_texts([question])[0]
        results = self.vector_store.search(query_embedding, self.top_k)
        retrieval_results = []
        for idx, score in results:
            if idx < 0 or score < self.similarity_threshold:
                continue
            chunk = self.vector_store.chunks[idx]
            retrieval_results.append(RetrievalResult(chunk=chunk, score=float(score)))
        if not retrieval_results:
            return "No relevant information found.", []
        context = self.format_context(retrieval_results)
        answer = self.generate_answer(question, context)
        citations = [
            {
                "document_id": r.chunk.document_id,
                "chunk_id": str(r.chunk.id),
                "text": r.chunk.text[:200],
            }
            for r in retrieval_results
        ]
        return answer, citations

    def format_context(self, results: List[RetrievalResult]) -> str:
        parts = []
        for result in results:
            part = (
                f"[Source: {result.chunk.document_id}, "
                f"Score: {result.score:.2f}]\n"
                f"{result.chunk.text.strip()}"
            )
            parts.append(part)
        return "\n\n".join(parts)

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are answering questions using only the context below.

Context:
{context}

Question:
{question}

Answer with citations using the provided source filenames.
"""
        payload = {
            "model": self.llm_model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. If you are not sure about the answer, say you don't know, and give the reasoning."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
        answer = data["message"]["content"].strip()
        return answer


# ----------------------
# In-memory storage (replace with DB in production)
# ----------------------

START_TIME = time.time()


# ----------------------
# RAG system
# ----------------------

rag_system = RAGSystem()

# ----------------------
# Middleware: request ID + logging
# ----------------------

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    logger.info(f"[{request_id}] Incoming request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"request_id": request_id, "error": "Internal server error"},
        )

    duration = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {duration:.3f}s")
    response.headers["X-Request-ID"] = request_id
    return response


# ----------------------
# Routes
# ----------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="ok",
        uptime_seconds=round(uptime, 2),
        version="1.0.0",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def query_rag(
    payload: QueryRequest,
    request: Request,
    api_key: str = Depends(authenticate),
    _: None = Depends(rate_limiter),
):
    request_id = request.state.request_id
    start_time = time.time()

    try:
        answer, citations = rag_system.query(payload.question)
    except Exception as e:
        logger.exception(f"[{request_id}] RAG query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query",
        )

    latency = time.time() - start_time
    return QueryResponse(
        request_id=request_id,
        answer=answer,
        citations=[Citation(**c) for c in citations],
        latency_seconds=round(latency, 4),
    )


@app.post(
    "/documents",
    status_code=201,
    responses={401: {"model": ErrorResponse}},
)
def add_document(
    payload: DocumentCreateRequest,
    request: Request,
    api_key: str = Depends(authenticate),
):
    request_id = request.state.request_id

    try:
        rag_system.add_document(payload.document_id, payload.text)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    logger.info(f"[{request_id}] Document added: {payload.document_id}")
    return {"document_id": payload.document_id, "created": True}


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def delete_document(
    document_id: str,
    request: Request,
    api_key: str = Depends(authenticate),
):
    request_id = request.state.request_id
    deleted = rag_system.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    logger.info(f"[{request_id}] Document deleted: {document_id}")
    return DocumentDeleteResponse(document_id=document_id, deleted=True)
