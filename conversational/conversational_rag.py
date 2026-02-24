import os
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

# Optional tokenizer for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None


# ---------------------------
# Logging Setup
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------
# LLM Configuration
# ---------------------------

MODEL_NAME = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------
# Data Structures
# ---------------------------

@dataclass
class Message:
    role: str
    content: str
    timestamp: float


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict


@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str
    embedding: np.ndarray
    metadata: Dict


@dataclass
class RetrievalResult:
    chunk_id: str
    parent_id: str
    text: str
    score: float
    metadata: Dict


# ---------------------------
# Token Utilities
# ---------------------------

def count_tokens(text: str, model: str = "gpt-4") -> int:
    if tiktoken is None:
        # Fallback: rough estimate
        return max(1, len(text.split()) // 0.75)
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ---------------------------
# LLM API Call
# ---------------------------

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    content = result["message"]["content"]
    prompt_tokens = result.get("prompt_eval_count")
    completion_tokens = result.get("eval_count")
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------
# Core Retrieval Components
# ---------------------------

class VectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: Dict[str, Chunk] = {}
        self.faiss_ids: List[str] = []

    def add_documents(self, documents: List[Document], chunk_size: int = 300, overlap: int = 50):
        chunks = []
        for doc in documents:
            words = doc.text.split()
            start = 0
            idx = 0
            while start < len(words):
                end = start + chunk_size
                text = " ".join(words[start:end])
                chunk_id = f"{doc.id}_chunk_{idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        parent_id=doc.id,
                        text=text,
                        embedding=None,
                        metadata=doc.metadata.copy()
                    )
                )
                start = end - overlap
                idx += 1

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self.chunks[chunk.id] = chunk

        self._build_index()

    def _build_index(self):
        dim = len(next(iter(self.chunks.values())).embedding)
        index = faiss.IndexFlatIP(dim)
        vectors = np.array([c.embedding for c in self.chunks.values()]).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self.index = index
        self.faiss_ids = list(self.chunks.keys())

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_emb = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            cid = self.faiss_ids[idx]
            chunk = self.chunks[cid]
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    parent_id=chunk.parent_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata
                )
            )
        return results


# ---------------------------
# Conversation Memory Manager
# ---------------------------

class ConversationManager:
    def __init__(
        self,
        session_id: str,
        max_buffer_turns: int = 6,
        max_token_budget: int = 4000,
        model_name: str = "gpt-4",
        storage_path: str = "sessions"
    ):
        self.session_id = session_id
        self.max_buffer_turns = max_buffer_turns
        self.max_token_budget = max_token_budget
        self.model_name = model_name
        self.storage_path = storage_path

        self.history: List[Message] = []
        self.summary: Optional[str] = None

        os.makedirs(storage_path, exist_ok=True)
        self.session_file = os.path.join(storage_path, f"{session_id}.json")
        self.load()

    # -----------------------
    # History Management
    # -----------------------

    def add_message(self, role: str, content: str):
        self.history.append(
            Message(role=role, content=content, timestamp=time.time())
        )
        self._trim_history()
        self.save()

    def _trim_history(self):
        # Sliding window
        if len(self.history) > self.max_buffer_turns:
            overflow = self.history[:-self.max_buffer_turns]
            self.history = self.history[-self.max_buffer_turns:]
            self._summarize_overflow(overflow)

    def _summarize_overflow(self, messages: List[Message]):
        text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        summary = self.simple_summarize(text)
        if self.summary:
            self.summary = f"{self.summary}\n{summary}"
        else:
            self.summary = summary

    def simple_summarize(self, text: str) -> str:
        # Placeholder summarizer. Replace with LLM call if desired.
        sentences = text.split(".")
        return ". ".join(sentences[:5]).strip()

    # -----------------------
    # Context Assembly
    # -----------------------

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Conversation summary:\n{self.summary}")
        for msg in self.history:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    def get_token_count(self) -> int:
        return count_tokens(self.get_context(), self.model_name)

    def ensure_token_budget(self):
        while self.get_token_count() > self.max_token_budget and self.history:
            overflow = self.history.pop(0)
            self._summarize_overflow([overflow])

    # -----------------------
    # Persistence
    # -----------------------

    def save(self):
        data = {
            "session_id": self.session_id,
            "summary": self.summary,
            "history": [asdict(m) for m in self.history]
        }
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.session_file):
            return
        with open(self.session_file, "r") as f:
            data = json.load(f)
            self.summary = data.get("summary")
            self.history = [Message(**m) for m in data.get("history", [])]


# ---------------------------
# Reference Resolution
# ---------------------------

class ReferenceResolver:
    def resolve(self, query: str, history: List[Message]) -> str:
        pronouns = ["it", "that", "this", "they", "those", "them"]
        tokens = query.lower().split()

        if not any(p in tokens for p in pronouns):
            return query

        # Look for last assistant or user message with a noun phrase
        for msg in reversed(history):
            if msg.role in ("assistant", "user"):
                candidate = self.extract_topic(msg.content)
                if candidate:
                    resolved = query
                    for p in pronouns:
                        resolved = resolved.replace(p, candidate)
                    return resolved

        return query

    def extract_topic(self, text: str) -> Optional[str]:
        words = text.split()
        if len(words) < 3:
            return None
        return " ".join(words[:5])


# ---------------------------
# Conversational RAG System
# ---------------------------

class ConversationalRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        conversation_manager: ConversationManager,
        resolver: ReferenceResolver,
        llm_stub: bool = False
    ):
        self.vector_store = vector_store
        self.conversation_manager = conversation_manager
        self.resolver = resolver
        self.llm_stub = llm_stub

    # -----------------------
    # Main Chat Loop Logic
    # -----------------------

    def ask(self, user_input: str, top_k: int = 5) -> str:
        # Resolve references using conversation history
        resolved_query = self.resolver.resolve(user_input, self.conversation_manager.history)

        # Retrieve relevant context
        results = self.vector_store.search(resolved_query, top_k=top_k)

        context = "\n\n".join(
            f"[Source: {r.parent_id}]\n{r.text}"
            for r in results
        )

        # Assemble prompt
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer is not in the context, say so clearly."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n{self.conversation_manager.get_context()}\n\n"
            f"User question:\n{resolved_query}"
        )

        # Generate response (stub or real LLM call)
        response = self.generate_response(full_prompt)

        # Save conversation
        self.conversation_manager.add_message("user", user_input)
        self.conversation_manager.add_message("assistant", response)

        return response

    def generate_response(self, prompt: str) -> str:
        if self.llm_stub:
            # Placeholder response logic
            return "This is a placeholder response based on the retrieved context."
        else:
            messages = [{"role": "user", "content": prompt}]
            result = call_llm(messages, temperature=0.7, max_tokens=512)
            return result["content"]


# ---------------------------
# CLI Interface
# ---------------------------

def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".md")):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            documents.append(
                Document(
                    id=fname,
                    text=text,
                    metadata={"source": fname, "type": "text"}
                )
            )
    return documents


def main():
    parser = argparse.ArgumentParser(description="Conversational RAG System")
    parser.add_argument("--docs", type=str, required=True, help="Path to documents folder")
    parser.add_argument("--session", type=str, default="default", help="Session ID")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Load documents and build vector store
    store = VectorStore()
    documents = load_documents_from_folder(args.docs)
    store.add_documents(documents)

    # Initialize conversation memory
    conversation_manager = ConversationManager(session_id=args.session)
    resolver = ReferenceResolver()

    rag = ConversationalRAG(
        vector_store=store,
        conversation_manager=conversation_manager,
        resolver=resolver,
    )

    print("Conversational RAG assistant. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = rag.ask(user_input, top_k=args.top_k)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
