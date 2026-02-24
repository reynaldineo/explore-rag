import os
import json
import time
import hashlib
import random
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import requests

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embed"
CACHE_DIR = "embedding_cache"
OUTPUT_DIR = "embedding_outputs_baru"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DATASET
# =========================

DATASET = [
    # Movies
    "A young wizard discovers his powers and attends a magical school.",
    "Two lovers from rival families fall in love in a tragic romance.",
    "A team of astronauts travel through a wormhole to save humanity.",
    "A detective hunts a serial killer in a dark, rainy city.",
    "A group of friends embark on a quest to destroy a powerful ring.",

    # Products
    "Wireless noise-cancelling over-ear headphones with Bluetooth.",
    "Compact mirrorless camera with 4K video recording.",
    "Smartphone with long battery life and fast charging.",
    "Mechanical keyboard with RGB lighting and hot-swappable switches.",
    "Running shoes designed for long-distance comfort.",

    # News
    "The central bank raises interest rates to fight inflation.",
    "A major tech company announces a breakthrough in AI research.",
    "The national team wins the championship after a dramatic final.",
    "Scientists discover a new species in the Amazon rainforest.",
    "The government unveils a new renewable energy policy.",

    # Code
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "public static void main(String[] args) { System.out.println(\"Hello World\"); }",
    "const sum = (a, b) => a + b;",
    "func add(a int, b int) int { return a + b }",
    "SELECT * FROM users WHERE email LIKE '%@example.com';",

    # Misc
    "A recipe for homemade chocolate chip cookies.",
    "Tips for improving productivity while working from home.",
    "Beginner's guide to investing in the stock market.",
    "How to train for a marathon safely and effectively.",
    "The history of ancient Rome and its emperors.",
]

# =========================
# UTILITY FUNCTIONS
# =========================

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

# =========================
# EMBEDDING FUNCTIONS
# =========================

def get_embedding(text: str, model: str = MODEL_NAME, use_cache: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    cache_key = f"{model}_{hash_text(text)}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            emb = np.array(data["embedding"])
            if emb.ndim > 1:
                emb = emb.flatten()
            return emb, data["metadata"]

    start_time = time.time()
    payload = {
        "model": model,
        "input": text
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    latency = time.time() - start_time

    embedding = data["embeddings"][0]

    metadata = {
        "model": model,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0),
        "latency_seconds": round(latency, 3),
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding, "metadata": metadata}, f, ensure_ascii=False)
    
    return np.array(embedding), metadata

def batch_get_embeddings(texts: List[str], model: str = MODEL_NAME, use_cache: bool = True) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    embeddings = []
    metadatas = []

    for text in texts:
        emb, meta = get_embedding(text, model=model, use_cache=use_cache)
        embeddings.append(emb)
        metadatas.append(meta)

    return embeddings, metadatas

# =========================
# SIMILARITY & SEARCH 
# =========================

def build_similarity_matric(embeddings: List[np.ndarray], metric: str = "cosine") -> np.ndarray:
    n = len(embeddings)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == "cosine":
                matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
            elif metric == "euclidean":
                matrix[i][j] = euclidean_distance(embeddings[i], embeddings[j])
            elif metric == "dot":
                matrix[i][j] = dot_product(embeddings[i], embeddings[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return matrix

def semantic_search(
    query: str,
    corpus_texts: List[str],
    corpus_embeddings: List[np.ndarray],
    top_k: int = 5,
    metric: str = "cosine"
) -> List[Dict[str, Any]]:
    query_embedding, _ = get_embedding(query)
    query_embedding = normalize(query_embedding)

    results = []
    for text, emb in zip(corpus_texts, corpus_embeddings):
        emb = normalize(emb)
        if metric == "cosine":
            score = cosine_similarity(query_embedding, emb)
        elif metric == "euclidean":
            score = euclidean_distance(query_embedding, emb)
        elif metric == "dot":
            score = dot_product(query_embedding, emb)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results.append({"text": text, "score": round(float(score), 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# =========================
# VISUALIZATION
# =========================

def visualize_embeddings_2d(embeddings: List[np.ndarray], texts: List[str], output_name: str = "embedding_plot"):
    embeddings_array = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

    for i, text in enumerate(texts):
        short_label = text[:30] + "..." if len(text) > 30 else text
        plt.annotate(short_label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.75)

    plt.title("2D Visualization of Text Embeddings (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")

# =========================
# PERFORMANCE BENCHMARKING
# =========================

def benchmark_embeddings(texts: List[str], model: str = MODEL_NAME) -> Dict[str, Any]:
    print("Running embedding performance benchmark...")
    start = time.time()
    embeddings, metadata_list = batch_get_embeddings(texts, model=model)
    total_time = time.time() - start

    total_prompt_tokens = sum(meta.get("prompt_tokens", 0) for meta in metadata_list)
    total_tokens = sum(meta.get("total_tokens", 0) for meta in metadata_list)

    return {
        "model": model,
        "num_texts": len(texts),
        "total_time_seconds": round(total_time, 3),
        "avg_time_per_text": round(total_time / len(texts), 4),
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
    }

# =========================
# MODEL COMPARISON HOOK
# =========================

def compare_models(texts: List[str], models: List[str]):
    results = []
    for model in models:
        print(f"\n🔍 Testing model: {model}")
        result = benchmark_embeddings(texts, model=model)
        results.append(result)
    return results

# =========================
# MAIN DRIVER
# =========================

def main():
    print("Starting Embedding Explorer\n")

    # 1. Get embeddings for the dataset
    embeddings, metadatas = batch_get_embeddings(DATASET)

    # 2. Build similarity matrix
    similarity_matrix = build_similarity_matric(embeddings, metric="cosine")

    matrix_path = os.path.join(OUTPUT_DIR, "similarity_matrix.json")
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(similarity_matrix.tolist(), f, ensure_ascii=False)
    print(f"Saved similarity matrix to {matrix_path}")

    # 3. visualize embeddings
    visualize_embeddings_2d(embeddings, DATASET, output_name="embedding_tsne_plot")

    # 4. Perform semantic search
    test_queries = [
        "wireless headphones",
        "romantic love story",
        "AI research breakthrough",
        "Python function for recursion",
        "renewable energy policy",
    ]

    search_results = {}
    for query in test_queries:
        results = semantic_search(query, DATASET, embeddings, top_k=5, metric="cosine")
        search_results[query] = results
        print(f"Query: {query}")
        for r in results:
            print(f"  • {r['score']:.3f} → {r['text']}")
        print()

    search_path = os.path.join(OUTPUT_DIR, "semantic_search_results.json")
    with open(search_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    print(f"Saved semantic search results to {search_path}")

    # 5. Benchmark embedding performance
    benchmark_result = benchmark_embeddings(DATASET)
    benchmark_path = os.path.join(OUTPUT_DIR, "embedding_benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, ensure_ascii=False, indent=2)
    print(f"Saved embedding benchmark results to {benchmark_path}")

    # 6️. Model comparison hook (optional)
    # Uncomment to compare multiple models
    # models_to_compare = [
    #     "text-embedding-3-small",
    #     "text-embedding-3-large",
    # ]
    # comparison_results = compare_models(DATASET, models_to_compare)
    # comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    # with open(comparison_path, "w", encoding="utf-8") as f:
    #     json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    # print(f"✅ Saved model comparison results to {comparison_path}")

if __name__ == "__main__":
    main()
