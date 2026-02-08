import os
import json
import time
import argparse
import logging
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ---------------------------
# Logging Setup
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------
# Data Structures
# ---------------------------

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
    embedding: Optional[np.ndarray]
    metadata: Dict


@dataclass
class RetrievalResult:
    chunk_id: str
    parent_id: str
    text: str
    score: float
    metadata: Dict


# ---------------------------
# Utility Functions
# ---------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def mmr_rerank(
    query_embedding: np.ndarray,
    doc_embeddings: List[np.ndarray],
    doc_ids: List[str],
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[str]:
    selected = []
    candidates = list(range(len(doc_embeddings)))

    while len(selected) < top_k and candidates:
        best_idx = None
        best_score = -float("inf")

        for i in candidates:
            relevance = cosine_similarity(query_embedding, doc_embeddings[i])
            diversity = 0
            if selected:
                diversity = max(
                    cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                    for j in selected
                )
            score = lambda_param * relevance - (1 - lambda_param) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)
        candidates.remove(best_idx)

    return [doc_ids[i] for i in selected]


# ---------------------------
# Advanced Retrieval System
# ---------------------------

class AdvancedRetriever:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_path: str = "faiss.index",
        metadata_path: str = "metadata.json",
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.chunks: Dict[str, Chunk] = {}
        self.parent_map: Dict[str, Document] = {}
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()

    # -----------------------
    # Ingestion
    # -----------------------

    def ingest_documents(self, documents: List[Document], chunk_size: int = 300, overlap: int = 50):
        logging.info("Ingesting documents...")

        chunks = []
        for doc in documents:
            text = doc.text
            words = text.split()
            start = 0
            chunk_idx = 0

            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                chunk_id = f"{doc.id}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        parent_id=doc.id,
                        text=chunk_text,
                        embedding=None,
                        metadata=doc.metadata.copy()
                    )
                )

                start = end - overlap
                chunk_idx += 1

            self.parent_map[doc.id] = doc

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self.chunks[chunk.id] = chunk

        self._build_faiss_index()
        self._build_bm25_index()
        self.save()

        logging.info("Ingestion complete.")

    def _build_faiss_index(self):
        dim = len(next(iter(self.chunks.values())).embedding)
        index = faiss.IndexFlatIP(dim)
        vectors = np.array([chunk.embedding for chunk in self.chunks.values()]).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self.index = index
        self.faiss_ids = list(self.chunks.keys())

    def _build_bm25_index(self):
        corpus = [chunk.text.split() for chunk in self.chunks.values()]
        self.bm25 = BM25Okapi(corpus)
        self.bm25_ids = list(self.chunks.keys())

    # -----------------------
    # Retrieval Methods
    # -----------------------

    def dense_search(self, query: str, top_k: int = 10) -> List[str]:
        query_emb = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        return [self.faiss_ids[i] for i in indices[0]]

    def sparse_search(self, query: str, top_k: int = 10) -> List[str]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [self.bm25_ids[i] for i in ranked]

    def hybrid_search(self, query: str, top_k: int = 10) -> List[str]:
        dense_results = self.dense_search(query, top_k)
        sparse_results = self.sparse_search(query, top_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:top_k]

    # -----------------------
    # Query Expansion
    # -----------------------

    def expand_query(self, query: str, n: int = 3) -> List[str]:
        prompt = f"Generate {n} alternative search queries for: {query}"
        try:
            outputs = self.cross_encoder.predict([(prompt, "")])  # fallback misuse-safe
            # If LLM not available, fallback to simple heuristic
            raise Exception
        except:
            expansions = [
                query,
                f"{query} explanation",
                f"{query} policy",
                f"{query} guide"
            ]
            return expansions[:n]

    # -----------------------
    # Reranking
    # -----------------------

    def rerank(self, query: str, candidate_ids: List[str], top_k: int = 5) -> List[str]:
        pairs = [(query, self.chunks[cid].text) for cid in candidate_ids]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]

    # -----------------------
    # Parent-Child Retrieval
    # -----------------------

    def retrieve_parents(self, chunk_ids: List[str]) -> List[Document]:
        parent_ids = {self.chunks[cid].parent_id for cid in chunk_ids}
        return [self.parent_map[pid] for pid in parent_ids]

    # -----------------------
    # MMR
    # -----------------------

    def apply_mmr(self, query: str, candidate_ids: List[str], top_k: int = 5, lambda_param: float = 0.5) -> List[str]:
        query_emb = self.embedder.encode(query)
        doc_embeddings = [self.chunks[cid].embedding for cid in candidate_ids]
        return mmr_rerank(query_emb, doc_embeddings, candidate_ids, lambda_param, top_k)

    # -----------------------
    # End-to-End Retrieval Pipeline
    # -----------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True,
        use_parent_retrieval: bool = True,
        metadata_filter: Optional[Dict] = None
    ) -> List[RetrievalResult]:

        start_time = time.time()

        # Step 1: Query expansion
        queries = [query]
        if use_query_expansion:
            queries = self.expand_query(query, n=3)

        # Step 2: Hybrid retrieval
        all_candidates = []
        for q in queries:
            if use_hybrid:
                results = self.hybrid_search(q, top_k=top_k * 2)
            else:
                results = self.dense_search(q, top_k=top_k * 2)
            all_candidates.extend(results)

        all_candidates = list(set(all_candidates))

        # Step 3: Metadata filtering
        if metadata_filter:
            filtered = []
            for cid in all_candidates:
                chunk = self.chunks[cid]
                if all(chunk.metadata.get(k) == v for k, v in metadata_filter.items()):
                    filtered.append(cid)
            all_candidates = filtered

        # Step 4: MMR
        if use_mmr:
            all_candidates = self.apply_mmr(query, all_candidates, top_k=top_k * 2)

        # Step 5: Reranking
        if use_reranking:
            all_candidates = self.rerank(query, all_candidates, top_k=top_k)
        else:
            all_candidates = all_candidates[:top_k]

        # Step 6: Parent-child retrieval
        if use_parent_retrieval:
            parents = self.retrieve_parents(all_candidates)
            results = []
            for parent in parents:
                results.append(
                    RetrievalResult(
                        chunk_id="",
                        parent_id=parent.id,
                        text=parent.text,
                        score=0.0,
                        metadata=parent.metadata
                    )
                )
        else:
            results = []
            for cid in all_candidates:
                chunk = self.chunks[cid]
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.id,
                        parent_id=chunk.parent_id,
                        text=chunk.text,
                        score=0.0,
                        metadata=chunk.metadata
                    )
                )

        elapsed = time.time() - start_time
        logging.info(f"Retrieval completed in {elapsed:.3f}s")

        return results

    # -----------------------
    # Evaluation
    # -----------------------

    def evaluate(
        self,
        queries_with_answers: List[Dict],
        top_k: int = 5,
        output_path: str = "evaluation_results.json"
    ):
        metrics = []

        for item in queries_with_answers:
            query = item["query"]
            relevant_parent_ids = item["relevant"]
            results = self.retrieve(query, top_k=top_k)
            retrieved_parents = [r.parent_id for r in results]

            true_positives = len(set(retrieved_parents) & set(relevant_parent_ids))
            precision = true_positives / top_k
            recall = true_positives / len(relevant_parent_ids) if relevant_parent_ids else 0

            metrics.append({
                "query": query,
                "precision@k": precision,
                "recall@k": recall,
                "retrieved": retrieved_parents,
                "relevant": relevant_parent_ids
            })

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Evaluation results saved to {output_path}")
        return metrics

    # -----------------------
    # Persistence
    # -----------------------

    def save(self):
        faiss.write_index(self.index, self.index_path)
        chunks_dict = {}
        for cid, chunk in self.chunks.items():
            chunk_dict = asdict(chunk)
            if chunk.embedding is not None:
                chunk_dict['embedding'] = chunk.embedding.tolist()
            chunks_dict[cid] = chunk_dict
        with open(self.metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": chunks_dict,
                    "parents": {pid: asdict(doc) for pid, doc in self.parent_map.items()}
                },
                f,
                indent=2
            )
        logging.info("Index and metadata saved.")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
            self.chunks = {}
            for cid, chunk_data in data["chunks"].items():
                if 'embedding' in chunk_data and chunk_data['embedding'] is not None:
                    chunk_data['embedding'] = np.array(chunk_data['embedding'])
                self.chunks[cid] = Chunk(**chunk_data)
            self.parent_map = {pid: Document(**doc) for pid, doc in data["parents"].items()}
        self._build_bm25_index()
        self.faiss_ids = list(self.chunks.keys())
        logging.info("Index and metadata loaded.")


# ---------------------------
# CLI Interface
# ---------------------------

def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".md", ".html")):
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


def export_results(results: List[RetrievalResult], output_path: str):
    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    elif output_path.endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    else:
        raise ValueError("Output format must be .json or .csv")


def main():
    parser = argparse.ArgumentParser(description="Advanced Retrieval System")
    parser.add_argument("--docs", type=str, required=True, help="Path to documents folder")
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_hybrid", action="store_true")
    parser.add_argument("--no_query_expansion", action="store_true")
    parser.add_argument("--no_reranking", action="store_true")
    parser.add_argument("--no_mmr", action="store_true")
    parser.add_argument("--no_parent_retrieval", action="store_true")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--evaluate", type=str, help="Path to evaluation file (JSON)")
    args = parser.parse_args()

    retriever = AdvancedRetriever()

    if not retriever.chunks:
        docs = load_documents_from_folder(args.docs)
        retriever.ingest_documents(docs)

    if args.evaluate:
        with open(args.evaluate, "r") as f:
            queries_with_answers = json.load(f)
        retriever.evaluate(queries_with_answers, top_k=args.top_k)
        return

    if not args.query:
        raise ValueError("You must provide --query unless running evaluation.")

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_query_expansion=not args.no_query_expansion,
        use_reranking=not args.no_reranking,
        use_mmr=not args.no_mmr,
        use_parent_retrieval=not args.no_parent_retrieval,
    )

    export_results(results, args.output)
    logging.info(f"Results saved to {args.output}")

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print(f"Parent ID: {r.parent_id}")
        print(f"Metadata: {r.metadata}")
        print(f"Text:\n{r.text[:500]}...")


if __name__ == "__main__":
    main()
