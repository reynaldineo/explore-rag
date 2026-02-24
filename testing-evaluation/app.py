import json
import argparse
import os
import time
import csv
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    raise ImportError("Please install nltk: pip install nltk")

import faiss
from tqdm import tqdm
import requests
from pathlib import Path

# -----------------------------
# Data Models
# -----------------------------

@dataclass
class GoldenExample:
    question: str
    expected_answer: str
    relevant_chunks: List[str]
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    retrieved_chunks: List[str]
    scores: List[float]


@dataclass
class GenerationResult:
    answer: str
    sources: List[str]
    latency_seconds: float
    token_count: int


@dataclass
class EvaluationResult:
    question: str
    difficulty: str
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    latency_seconds: float
    token_count: int
    passed: bool
    failure_reason: str = ""


# -----------------------------
# Utility Functions
# -----------------------------

def load_golden_dataset(path: str) -> List[GoldenExample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = []
    for item in raw:
        dataset.append(GoldenExample(
            question=item["question"],
            expected_answer=item["expected_answer"],
            relevant_chunks=item.get("relevant_chunks", []),
            difficulty=item.get("difficulty", "medium"),
            metadata=item.get("metadata", {})
        ))
    return dataset


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Retrieval Metrics
# -----------------------------

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    def dcg(scores):
        return sum(
            (2 ** rel - 1) / math.log2(idx + 2)
            for idx, rel in enumerate(scores)
        )

    relevance_map = {chunk: 1 for chunk in relevant}
    actual_scores = [relevance_map.get(r, 0) for r in retrieved[:k]]
    ideal_scores = sorted(actual_scores, reverse=True)

    dcg_val = dcg(actual_scores)
    idcg_val = dcg(ideal_scores)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# -----------------------------
# Generation Metrics
# -----------------------------

def exact_match(predicted: str, expected: str) -> float:
    return float(predicted.strip().lower() == expected.strip().lower())


def rouge_l(predicted: str, expected: str) -> float:
    # Simple ROUGE-L using longest common subsequence ratio
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    pred_tokens = predicted.split()
    exp_tokens = expected.split()
    if not exp_tokens:
        return 0.0
    lcs_len = lcs(pred_tokens, exp_tokens)
    return lcs_len / len(exp_tokens)


def bleu_score(predicted: str, expected: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [expected.split()],
        predicted.split(),
        smoothing_function=smoothie
    )


class SemanticSimilarityScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def score(self, predicted: str, expected: str) -> float:
        texts = [predicted, expected]
        tfidf = self.vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)


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
                    print(f"Failed to load {file_path}: {e}")
        return documents

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
                    source=document.source,
                    metadata=document.metadata.copy(),
                )
            )
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

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
            norm = np.linalg.norm(emb)
            emb = emb / norm if norm > 0 else emb
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

# -----------------------------
# RAG System Adapter
# -----------------------------

class RAGSystem:
    """
    Replace this class with your actual RAG implementation.
    This adapter defines the contract used by the evaluator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def retrieve(self, question: str) -> RetrievalResult:
        """
        Must return:
        - retrieved_chunks: list of chunk IDs or strings
        - scores: similarity scores (same length as retrieved_chunks)
        """
        raise NotImplementedError("Implement retrieval()")

    def generate(self, question: str, retrieved_chunks: List[str]) -> GenerationResult:
        """
        Must return:
        - answer: generated answer
        - sources: chunk IDs or source identifiers used
        - latency_seconds: float
        - token_count: int
        """
        raise NotImplementedError("Implement generate()")


# -----------------------------
# Evaluator
# -----------------------------

class RAGEvaluator:
    def __init__(self, rag_system: RAGSystem, k: int = 5):
        self.rag = rag_system
        self.k = k
        self.semantic_scorer = SemanticSimilarityScorer()

    def evaluate_example(self, example: GoldenExample) -> EvaluationResult:
        start_time = time.time()

        retrieval = self.rag.retrieve(example.question)
        generation = self.rag.generate(example.question, retrieval.retrieved_chunks)

        latency = generation.latency_seconds
        tokens = generation.token_count

        retrieval_metrics = {
            "precision@k": precision_at_k(retrieval.retrieved_chunks, example.relevant_chunks, self.k),
            "recall@k": recall_at_k(retrieval.retrieved_chunks, example.relevant_chunks, self.k),
            "mrr": mean_reciprocal_rank(retrieval.retrieved_chunks, example.relevant_chunks),
            "ndcg@k": ndcg_at_k(retrieval.retrieved_chunks, example.relevant_chunks, self.k),
        }

        generation_metrics = {
            "exact_match": exact_match(generation.answer, example.expected_answer),
            "rouge_l": rouge_l(generation.answer, example.expected_answer),
            "bleu": bleu_score(generation.answer, example.expected_answer),
            "semantic_similarity": self.semantic_scorer.score(generation.answer, example.expected_answer),
        }

        passed = generation_metrics["semantic_similarity"] >= 0.4

        failure_reason = ""
        if not passed:
            if retrieval_metrics["recall@k"] == 0:
                failure_reason = "Retrieval failure"
            else:
                failure_reason = "Generation failure"

        return EvaluationResult(
            question=example.question,
            difficulty=example.difficulty,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_seconds=latency,
            token_count=tokens,
            passed=passed,
            failure_reason=failure_reason
        )

    def evaluate_dataset(self, dataset: List[GoldenExample]) -> List[EvaluationResult]:
        results = []
        for example in dataset:
            result = self.evaluate_example(example)
            results.append(result)
        return results


# -----------------------------
# Regression Testing
# -----------------------------

class RegressionTester:
    def __init__(self, baseline_path: str, tolerance: float = 0.05):
        self.baseline = self._load_baseline(baseline_path)
        self.tolerance = tolerance

    def _load_baseline(self, path: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        results = {}
        for metric, baseline_value in self.baseline.items():
            current_value = current_metrics.get(metric, 0.0)
            delta = current_value - baseline_value
            results[metric] = delta >= -self.tolerance
        return results


# -----------------------------
# Parameter Sweep
# -----------------------------

class ParameterSweepRunner:
    def __init__(self, base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]):
        self.base_config = base_config
        self.param_grid = param_grid

    def run(self, dataset: List[GoldenExample], evaluator_factory) -> List[Dict[str, Any]]:
        from itertools import product

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        results = []

        for combo in product(*values):
            config = dict(self.base_config)
            config.update(dict(zip(keys, combo)))

            rag_system = evaluator_factory(config)
            evaluator = RAGEvaluator(rag_system, k=config.get("k", 5))
            eval_results = evaluator.evaluate_dataset(dataset)

            avg_semantic = np.mean([r.generation_metrics["semantic_similarity"] for r in eval_results])
            avg_recall = np.mean([r.retrieval_metrics["recall@k"] for r in eval_results])
            avg_latency = np.mean([r.latency_seconds for r in eval_results])

            result_row = {
                **config,
                "avg_semantic_similarity": avg_semantic,
                "avg_recall@k": avg_recall,
                "avg_latency_seconds": avg_latency
            }
            results.append(result_row)

        return results


# -----------------------------
# Report Generation
# -----------------------------

class ReportGenerator:
    def __init__(self, results: List[EvaluationResult]):
        self.results = results

    def summary_metrics(self) -> Dict[str, float]:
        metrics = defaultdict(list)
        for r in self.results:
            for k, v in r.retrieval_metrics.items():
                metrics[k].append(v)
            for k, v in r.generation_metrics.items():
                metrics[k].append(v)
            metrics["latency_seconds"].append(r.latency_seconds)
            metrics["token_count"].append(r.token_count)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def failure_analysis(self) -> Dict[str, int]:
        failures = defaultdict(int)
        for r in self.results:
            if not r.passed:
                failures[r.failure_reason] += 1
        return dict(failures)

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": self.summary_metrics(),
            "failure_analysis": self.failure_analysis(),
            "results": [
                {
                    "question": r.question,
                    "difficulty": r.difficulty,
                    "retrieval_metrics": r.retrieval_metrics,
                    "generation_metrics": r.generation_metrics,
                    "latency_seconds": r.latency_seconds,
                    "token_count": r.token_count,
                    "passed": r.passed,
                    "failure_reason": r.failure_reason,
                }
                for r in self.results
            ]
        }


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--dataset", type=str, required=True, help="Path to golden dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="evaluation_reports", help="Directory to save reports")
    parser.add_argument("--k", type=int, default=5, help="Top-K retrieval cutoff")
    parser.add_argument("--baseline", type=str, help="Path to baseline metrics JSON for regression testing")
    parser.add_argument("--run_sweep", action="store_true", help="Run parameter sweep")
    return parser.parse_args()


# -----------------------------
# Example RAG System (Stub)
# -----------------------------

class ExampleRAGSystem(RAGSystem):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docs_folder = config.get("docs_folder", "../docs")
        self.embedding_model = EmbeddingModel("nomic-embed-text")
        self.chunker = Chunker(500, 100)
        self.vector_store = None
        self.ingest()

    def ingest(self):
        loader = DocumentLoader()
        documents = loader.load_documents(self.docs_folder)
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError("No documents found to ingest.")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(chunks)

    def retrieve(self, question: str) -> RetrievalResult:
        query_embedding = self.embedding_model.embed_texts([question])[0]
        results = self.vector_store.search(query_embedding, self.config.get("k", 5))
        retrieved_chunks = []
        scores = []
        for idx, score in results:
            if idx >= 0:
                chunk = self.vector_store.chunks[idx]
                retrieved_chunks.append(str(chunk.id))
                scores.append(float(score))
        return RetrievalResult(retrieved_chunks=retrieved_chunks, scores=scores)

    def generate(self, question: str, retrieved_chunks: List[str]) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(answer="No context provided.", sources=[], latency_seconds=0.0, token_count=0)
        context_parts = []
        for chunk_id_str in retrieved_chunks:
            chunk_id = int(chunk_id_str)
            chunk = self.vector_store.chunks[chunk_id]
            context_parts.append(chunk.text)
        context = "\n\n".join(context_parts)
        prompt = f"""
You are answering questions using only the context below.

Context:
{context}

Question:
{question}

Provide a concise answer based only on the context. Do not add extra information or citations.
"""
        start_time = time.time()
        payload = {
            "model": "llama3.2",
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
            print(f"Error: {e}")
            return GenerationResult(answer="Error generating answer.", sources=[], latency_seconds=time.time() - start_time, token_count=0)
        latency = time.time() - start_time
        answer = data["message"]["content"].strip()
        token_count = len(answer.split())
        sources = retrieved_chunks[:2]
        return GenerationResult(answer=answer, sources=sources, latency_seconds=latency, token_count=token_count)


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_golden_dataset(args.dataset)

    rag_system = ExampleRAGSystem(config={"k": args.k})
    evaluator = RAGEvaluator(rag_system, k=args.k)

    results = evaluator.evaluate_dataset(dataset)

    report = ReportGenerator(results)
    report_data = report.to_json()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.json")
    save_json(json_path, report_data)

    csv_rows = []
    for r in results:
        row = {
            "question": r.question,
            "difficulty": r.difficulty,
            **r.retrieval_metrics,
            **r.generation_metrics,
            "latency_seconds": r.latency_seconds,
            "token_count": r.token_count,
            "passed": r.passed,
            "failure_reason": r.failure_reason,
        }
        csv_rows.append(row)

    csv_path = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.csv")
    save_csv(csv_path, csv_rows)

    print(f"Evaluation completed.")
    print(f"JSON report saved to: {json_path}")
    print(f"CSV report saved to: {csv_path}")

    if args.baseline:
        tester = RegressionTester(args.baseline)
        summary = report.summary_metrics()
        regression_results = tester.compare(summary)
        print("Regression test results:")
        for metric, passed in regression_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {metric}: {status}")

    if args.run_sweep:
        print("Running parameter sweep...")
        base_config = {"k": args.k}
        param_grid = {
            "k": [1, 3, 5, 10],
        }

        def evaluator_factory(config):
            return ExampleRAGSystem(config=config)

        sweep_runner = ParameterSweepRunner(base_config, param_grid)
        sweep_results = sweep_runner.run(dataset, evaluator_factory)

        sweep_path = os.path.join(args.output_dir, f"parameter_sweep_{timestamp}.json")
        save_json(sweep_path, sweep_results)
        print(f"Parameter sweep results saved to: {sweep_path}")


if __name__ == "__main__":
    main()
