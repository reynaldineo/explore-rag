---
title: 'Beyond Basic Vector Search: Advanced RAG Retrieval Techniques'
description: A practical guide to improving retrieval quality using hybrid search, query expansion, reranking, and diversity-aware ranking.
date: 2030-01-18
tags: ['RAG', 'Information Retrieval', 'LLM', 'Search Systems']
published: true
---

# Beyond Basic Vector Search: Advanced RAG Retrieval Techniques

> A practical guide to improving retrieval quality using hybrid search, query expansion, reranking, and diversity-aware ranking.

Retrieval quality determines how useful a retrieval-augmented generation systems can be. If the retrieved context is weak, incomplete, or misleading, the generation step will fail regardless of how good the language model is.

Simple similarity search works well for many cases, but it breaks down on ambiguous queries, long documents, and mixed-content corpora. This post walks through several retrieval techniques that go beyond basic vector search and shows how they improve recall, precision, and answer grounding in real systems.

The focus is practical. Each technique is explained in terms of what problem it solves, how it works, and how it fits into a production retrieval pipeline.

## Why Simple Similarity Fails

Dense vector similarity assumes that semantic closeness is enough. In practice, this often misses key signals.

For example, consider the query:

> "What is the refund policy for enterprise customers?"

A vector search may retrieve general pricing documents or consumer refund policies, even if a specific enterprise policy exists. This happens because semantic similarity does not account for term-level importance, document structure, or metadata constraints.

Other common failure cases include:

- Queries with rare keywords or product codes
- Ambiguous questions with multiple plausible interpretations
- Long documents where relevant information is buried in a small section
- Repetitive content where top-K results are redundant

These limitations motivate more structured retrieval strategies.

## Hybrid Search Explained

Hybrid search combines dense vector retrieval with sparse keyword-based retrieval, typically BM25. Each method captures different signals.

Dense retrieval:

- Captures semantic meaning
- Works well for paraphrases and conceptual similarity
- Struggles with rare terms and exact matches

Sparse retrieval (BM25):

- Captures keyword overlap and term importance
- Works well for names, IDs, and structured text
- Struggles with paraphrasing and implicit meaning

By combining both, the system benefits from semantic coverage and lexical precision.

## Rank Fusion

A common approach is rank fusion. Each retriever produces a ranked list, and there lists are merged into single ranking.

A simple fusion method assigns a score based on rnak posititon:

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

This method rewards documents that appear high in multiple rankings, even if neither retriever ranked them first individually.

Hybrid retrieval typically improves recall at the cost of some additional complexity.

## Query Expansion

User queries are often underspecified. A single sentence may not fully express the intent or the relevant terminology.

Query expansion addresses this by generating multiple variations of the original query. These variants are then used for retrieval, and the results are merged.

For example, the query:

> How does billing work for enterprise customers?

Could be expanded to:

- "Enterprise billing process and invoicing"
- "How enterprise pricing and payments are handled"
- "Billing policies for enterprise accounts"

These variants increase the chance of matching relevant documents that use different wording.

## LLM-based Expansion

A language model can generate controlled variations:

```python
def expand_query(llm, query, n=3):
    prompt = f"Generate {n} alternative search queries for: {query}"
    response = llm.generate(prompt)
    return response.split("\n")
```

The expanded queries are treated as independent retrieval queries. Their results are then combined using rank fusion or score aggregation.

Query expansion improves recall, especially for ambiguous or high-level questions, but it increases retrieval cost and latency.

## Reranking with Cross-Encoders

Initial retrieval typically uses fast models such as bi-encoders or sparse scoring. These models score documents independently of the query in a coarse way.

Reranking introduces a slower but more accurate model that jointly encodes the query and each candidate document. This allows the model to evaluate fine-grained relevance.

How It Works:

1. Retrieve top-K candidates using fast retrieval.
2. Pass each query-document pair through a cross-encoder.
3. Sort candidates by cross-encoder score.

Example:

```python
def rerank(cross_encoder, query, candidates):
    pairs = [(query, doc.text) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

Cross-encoders significantly improve precision at top ranks. This is especially useful when only a small number of retrieved chunks are passed to the generation model.

The tradeoff is latency. Cross-encoders are computationally expensive and should be applied only to a limited candidate set.

## Parent-Child Retrieval

Chunking improves retrieval granularity but can remove important context. A small chunk may match the query, but it may not contain enough surrounding information to answer it fully.

Parent-child retrieval addresses this by indexing small chunks but returning larger parent units at retrieval time.

Example

- Parent unit: full section or paragraph
- Child unit: sentence-level or small chunk embeddings

Workflow:

1. Index child chunks with references to their parent.
2. Retrieve top-K child chunks.
3. Return their corresponding parent documents or paragraphs.

```python
def retrieve_parent(child_results, parent_map):
    parents = []
    for child in child_results:
        parent_id = parent_map[child.id]
        parents.append(parent_id)
    return list(set(parents))
```

This approach preserves retrieval precision while restoring context for generation. It is particularly useful for long technical documents and structured manuals.

## Metadata Filtering

Not all documents are equally relevant for every query. Metadata filtering allows the system to pre-filter candidates before retrieval.

Common metadata fields include:

- Document type
- Publication date
- Source system
- Access level
- Product or customer segment

Example:

```python
def filter_by_metadata(chunks, filters):
    return [
        chunk for chunk in chunks
        if all(chunk.metadata.get(k) == v for k, v in filters.items())
    ]
```

Metadata filtering reduces noise, improves precision, and can significantly lower retrieval latency by shrinking the search space.

This technique is often combined with hybrid retrieval and reranking.

## MMR: Maximal Marginal Relevance

Simple top-K retrieval often returns highly similar chunks. This leads to redundancy and limits coverage of different aspects of a query.

MMR balances relevance with diversity. It selects documents that are both relevant to the query and dissimilar to each other.

Algorithm:
At each step, select the document that maximizes:

```ini
MMR = λ * relevance(query, doc) - (1 - λ) * max_similarity(doc, selected_docs)
```

Example implementation:

```python
def mmr(query_embedding, doc_embeddings, lambda_param=0.5, top_k=5):
    selected = []
    candidates = list(range(len(doc_embeddings)))

    while len(selected) < top_k and candidates:
        best = None
        best_score = -float("inf")
        for i in candidates:
            relevance = cosine_similarity(query_embedding, doc_embeddings[i])
            diversity = max(
                cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                for j in selected
            ) if selected else 0
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)
        candidates.remove(best)

    return selected
```

MMR improves coverage, especially for analytical and comparative questions, at the cost of slightly lower average relevance per chunk.

## Performance Impact

Each advanced technique improves retrieval quality but adds latency. Understanding these tradeoffs is essential for production systems.

| Technique          | Quality Impact   | Latency Impact | Typical Use Case                   |
| ------------------ | ---------------- | -------------- | ---------------------------------- |
| Hybrid search      | High recall      | Low to medium  | General-purpose retrieval          |
| Query expansion    | Higher recall    | Medium         | Ambiguous queries                  |
| Reranking          | High precision   | Medium to high | When top results must be correct   |
| Parent-child       | Better context   | Low            | Long or structured documents       |
| Metadata filtering | Higher precision | Low            | Domain-constrained retrieval       |
| MMR                | Better diversity | Low to medium  | Exploratory and analytical queries |


Combining multiple techniques often yields the best results. For example, hybrid search followed by reranking and MMR can provide high-quality, diverse results suitable for complex queries.

## Conclusion

Advanced retrieval techniques address the practical gaps left by simple vector search. They improve recall, precision, context coverage, and answer grounding.

This makes it clear that retrieval is not a single algorithm but a pipeline of decisions. Each stage shapes what information reaches the language model.

Well-designed retrieval systems do not aim for theoretical optimality. They aim for predictable, measurable improvements under real constraints. This requires testing, instrumentation, and careful tradeoff analysis, not just stronger models.