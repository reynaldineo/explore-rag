import os
import sys
import argparse
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
import requests


# ---------------------------
# Utility Functions
# ---------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------
# Configuration
# ---------------------------

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3
LOG_FILE = "rag_mvp.log"


# ---------------------------
# Logging Setup
# ---------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class Document:
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    metadata: Dict[str, str]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


# ---------------------------
# Document Loader
# ---------------------------

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".csv"}

    def load_documents(self, folder_path: str) -> List[Document]:
        documents = []
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")

        for file_path in folder.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    documents.append(
                        Document(
                            text=text,
                            source=str(file_path),
                            metadata={"file_name": file_path.name},
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {folder_path}")
        return documents


# ---------------------------
# Chunking
# ---------------------------

class Chunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    id=-1,  # Will be assigned later
                    text=chunk_text,
                    source=document.source,
                    metadata=document.metadata.copy(),
                )
            )
            start += self.chunk_size - self.overlap
            chunk_id += 1

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# ---------------------------
# Embedding Model
# ---------------------------

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


# ---------------------------
# Vector Store (FAISS)
# ---------------------------

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
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        return list(zip(indices[0], scores[0]))

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                    }
                    for chunk in self.chunks
                ],
                f,
                indent=2,
            )
        logger.info("Vector store saved")

    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
            self.chunks = [
                Chunk(
                    id=item["id"],
                    text=item["text"],
                    source=item["source"],
                    metadata=item["metadata"],
                )
                for item in chunk_data
            ]
        logger.info("Vector store loaded")


# ---------------------------
# RAG System
# ---------------------------

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.llm_model_name = llm_model_name
        self.chunker = Chunker(chunk_size, overlap)
        self.vector_store: Optional[VectorStore] = None
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def ingest(self, folder_path: str):
        loader = DocumentLoader()
        documents = loader.load_documents(folder_path)

        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError("No documents found to ingest.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(chunks)

    def retrieve(self, query: str) -> List[RetrievalResult]:
        query_embedding = self.embedding_model.embed_texts([query])[0]
        results = self.vector_store.search(query_embedding, self.top_k)

        retrieval_results = []
        for idx, score in results:
            if idx < 0:
                continue
            if score < self.similarity_threshold:
                continue
            chunk = self.vector_store.chunks[idx]
            retrieval_results.append(RetrievalResult(chunk=chunk, score=float(score)))

        logger.info(
            f"Retrieved {len(retrieval_results)} chunks for query: {query}"
        )
        return retrieval_results

    def format_context(self, results: List[RetrievalResult]) -> str:
        parts = []
        for result in results:
            part = (
                f"[Source: {os.path.basename(result.chunk.source)}, "
                f"Score: {result.score:.2f}]\n"
                f"{result.chunk.text.strip()}"
            )
            parts.append(part)
        return "\n\n".join(parts)

    def generate_answer(self, question: str, context: str) -> Tuple[str, int]:
        prompt = f"""
You are answering questions using only the context below.

Context:
{context}

Question:
{question}

Answer with citations using the provided source filenames.
"""

        start_time = time.time()

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
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise

        latency = time.time() - start_time
        answer = data["message"]["content"].strip()

        logger.info(f"LLM response time: {latency:.2f}s")
        return answer, latency

    def answer_question(self, question: str) -> Dict:
        start_time = time.time()

        retrieval_results = self.retrieve(question)
        if not retrieval_results:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "sources": [],
                "latency": time.time() - start_time,
            }

        context = self.format_context(retrieval_results)
        answer, generation_latency = self.generate_answer(question, context)

        total_latency = time.time() - start_time
        sources = list({os.path.basename(r.chunk.source) for r in retrieval_results})

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency": total_latency,
            "retrieval_latency": total_latency - generation_latency,
            "generation_latency": generation_latency,
        }

        logger.info(f"Answered question: {question}")
        return result


# ---------------------------
# CLI Interface
# ---------------------------

def run_cli():
    parser = argparse.ArgumentParser(description="Minimal RAG System CLI")
    parser.add_argument("--input", type=str, required=True, help="Folder containing documents")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--save-index", type=str, default=None, help="Path to save FAISS index")
    parser.add_argument("--load-index", type=str, default=None, help="Path to load FAISS index")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to metadata JSON")
    args = parser.parse_args()

    rag = RAGSystem(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    if args.load_index and args.metadata_path:
        rag.vector_store = VectorStore(embedding_dim=768)  # For nomic-embed-text
        rag.vector_store.load(args.load_index, args.metadata_path)
    else:
        rag.ingest(args.input)
        if args.save_index and args.metadata_path:
            rag.vector_store.save(args.save_index, args.metadata_path)

    print("\nRAG system ready. Type your question (or 'exit' to quit).\n")

    while True:
        question = input(">> ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = rag.answer_question(question)

        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src}")
        print(f"\nLatency: {result['latency']:.2f}s\n")

        logger.info(f"Session result: {json.dumps(result, indent=2)}")


# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Error: {e}")
