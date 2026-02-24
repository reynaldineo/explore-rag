import os
import json
import time 
import random
import string 
import hashlib 
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import requests

# =========================
# CONFIGURATION
# =========================

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embed"
CACHE_DIR = "embedding_cache"
OUTPUT_DIR = "vector_db_outputs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# EMBEDDING UTILS
# =========================

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_embedding(text: str, model: str = EMBEDDING_MODEL, use_cache: bool = True) -> np.ndarray:
    cache_key = f"{model}_{hash_text(text)}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return np.array(data["embedding"])

    payload = {
        "model": model,
        "input": text
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    embedding = data["embeddings"][0]

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding}, f, ensure_ascii=False)

    return np.array(embedding)

def batch_get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[np.ndarray]:
    embeddings = []
    for text in texts:
        emb = get_embedding(text, model=model)
        embeddings.append(emb)
    return embeddings

# =========================
# SYNTHETIC DATA GENERATOR
# =========================

def generate_synthetic_documents(n: int = 1000) -> List[Dict[str, Any]]:
    categories = ["news", "product", "blog", "research", "tutorial"]
    sources = ["website", "email", "report", "chat", "pdf"]

    documents = []
    for i in range(n):
        text = f"Document {i}: " + " ".join(random.choices(string.ascii_lowercase, k=100))
        metadata = {
            "id": i,
            "source": random.choice(sources),   
            "category": random.choice(categories),
            "date": f"202{random.randint(0, 5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        }
        documents.append({"text": text, "metadata": metadata})
    return documents

# =========================
# VECTOR DATABASE CLASS
# =========================

class VectorDatabase:
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 100,
        m: int = 32,
        ef_construction: int = 200
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.ef_construction = ef_construction
        
        self.index = self._build_index()
        self.metadata_store: List[Dict[str, Any]] = []
        self.text_store: List[str] = []

    def _build_index(self):
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, self.m)
            index.hnsw.efConstruction = self.ef_construction
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index

    def train(self, embeddings: np.ndarray):
        if self.index_type == "ivf":
            print("Training IVF index...")
            self.index.train(embeddings)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = batch_get_embeddings(texts)
        embeddings = np.array([normalize(emb) for emb in embeddings]).astype("float32")

        if self.index_type == "ivf" and not self.index.is_trained:
            self.train(embeddings)
        
        self.index.add(embeddings)
        self.metadata_store.extend(metadatas)
        self.text_store.extend(texts)
    
    def delete(self, ids: List[int]):
        selector = faiss.IDSelectorArray(len(ids), np.array(ids, dtype=np.int64))
        self.index.remove_ids(selector)
        # Note: Metadata and text store cleanup is not handled here for simplicity.
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        nprobe: int = 10,
        ef_search: int = 50,
    ) -> List[Dict[str, Any]]:
        query_emb = normalize(get_embedding(query)).astype("float32").reshape(1, -1)

        if self.index_type == "ivf":
            self.index.nprobe = nprobe
        elif self.index_type == "hnsw":
            self.index.hnsw.efSearch = ef_search
        
        scores, indices = self.index.search(query_emb, top_k * 5)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            metadata = self.metadata_store[idx]
            text = self.text_store[idx]

            if filters:
                passed = all(metadata.get(k) == v for k, v in filters.items())
                if not passed:
                    continue

            results.append({
                "id": idx,
                "score": float(score),
                "text": text,
                "metadata": metadata
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata_store": self.metadata_store,
                "text_store": self.text_store,
                "config": {
                    "dim": self.dimension,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "m": self.m,
                    "ef_construction": self.ef_construction
                },
            }, f, ensure_ascii=False, indent=2)

        print(f"Saved index to {path}")
    
    @classmethod
    def load(cls, path: str):
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "metadata.json")

        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = data["config"] 
        db = cls(
            dimension=config["dim"],
            index_type=config["index_type"],
            nlist=config["nlist"],
            m=config["m"],
            ef_construction=config["ef_construction"]
        )
        db.index = index
        db.metadata_store = data["metadata_store"]
        db.text_store = data["text_store"]

        print(f"Loaded index from {path}")
        return db
    
# =========================
# BENCHMARKING FUNCTION
# =========================

def benchmark_search(
    db: VectorDatabase,
    queries: List[str],
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    nprobe: int = 10,
    ef_search: int = 50,
) -> Dict[str, Any]:
    latencies = []
    for query in queries:
        start_time = time.time()
        _ = db.search(query=query, top_k=top_k, filters=filters, nprobe=nprobe, ef_search=ef_search)
        latencies.append(time.time() - start_time)

    return {
        "avg_latency_ms": round(1000 * sum(latencies) / len(latencies), 3),
        "p95_latency_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 3),
    }

def compute_recall(
    flat_db: VectorDatabase,
    ann_db: VectorDatabase,
    queries: List[str],
    top_k: int = 5,
) -> float:
    recalls = []

    for query in queries:
        flat_results = flat_db.search(query=query, top_k=top_k)
        ann_results = ann_db.search(query=query, top_k=top_k)

        flat_ids = set(res["id"] for res in flat_results)
        ann_ids = set(res["id"] for res in ann_results)

        intersection = flat_ids.intersection(ann_ids)
        recall = len(intersection) / len(flat_ids) if flat_ids else 0.0
        recalls.append(recall)
    
    return round(sum(recalls) / len(recalls), 4) if recalls else 0.0

def benchmark_index_build(db: VectorDatabase, texts: List[str], metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
    start_time = time.time()
    db.add_documents(texts, metadatas)
    build_time = time.time() - start_time

    index_size_bytes = faiss.get_mem_usage_kb() * 1024 # Convert KB to Bytes

    return {
        "build_time_seconds": round(build_time, 3),
        "index_memory_bytes": index_size_bytes,
    }

# =========================
# MAIN DRIVER
# =========================

def main():
    print("Starting vector database demo...")

    # 1. Generate synthetic documents
    num_documents = 5000
    documents = generate_synthetic_documents(num_documents)
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # 2. Initialize vector databases
    sample_embedding = get_embedding(texts[0])
    dim = len(sample_embedding)

    flat_db = VectorDatabase(dimension=dim, index_type="flat")
    ivf_db = VectorDatabase(dimension=dim, index_type="ivf", nlist=100)
    hnsw_db = VectorDatabase(dimension=dim, index_type="hnsw", m=32)

    # 3. Benchmark index building
    build_results = {}

    for name, db in [("flat", flat_db), ("ivf", ivf_db), ("hnsw", hnsw_db)]:
        print(f"Building {name.upper()} index...")
        result = benchmark_index_build(db, texts, metadatas)
        build_results[name] = result
        print(f"{name} index built in {result['build_time_seconds']} seconds, size: {result['index_memory_bytes']} bytes")
    
    # 4. Prepare benchmark queries
    test_queries = random.sample(texts, 50)

    search_results = {}
    search_results["flat"] = benchmark_search(flat_db, test_queries)
    search_results["ivf"] = benchmark_search(ivf_db, test_queries, nprobe=10)
    search_results["hnsw"] = benchmark_search(hnsw_db, test_queries, ef_search=50)

    # 5. Measure recall accuracy
    recall_results = {
        "ivf_vs_flat": compute_recall(flat_db, ivf_db, test_queries),
        "hnsw_vs_flat": compute_recall(flat_db, hnsw_db, test_queries),
    }

    # 6. Test metadata filtering
    sample_filter = {"category": "news"}
    filter_results = {
        "flat_filtered": benchmark_search(flat_db, test_queries, filters=sample_filter),
        "ivf_filtered": benchmark_search(ivf_db, test_queries, filters=sample_filter),
        "hnsw_filtered": benchmark_search(hnsw_db, test_queries, filters=sample_filter),
    }

    # 7. Output results
    output = {
        "build_results": build_results,
        "search_results": search_results,
        "recall_results": recall_results,
        "filter_results": filter_results,
    }

    output_path = os.path.join(OUTPUT_DIR, "vector_db_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Benchmark results saved to {output_path}")

if __name__ == "__main__":
    main()