# Exploring How Semantic Search Works: Understanding Embeddings and Vector Similarity

> A deep dive into semantic search, exploring how embeddings and vector similarity enable more intuitive and relevant information retrieval.

Modern search does more than match keywords. It tries to understand the meaning of text, and **embeddings** are what make this possible.

This post shows how text is turned into vectors that capture meaning. It explains how similarity between these vectors is measured, and how this enables semantic search.

By the end, you will understand how meaning is captured, how related content is found, and how these parts work together in a search system.

## From Text to Vectors

At a basic level, embeddings turn text into numbers. These numbers are not random. Each piece of text becomes a vector that represents its meaning rather than its exact wording.

For example, instead of treating this as plain text:

```
"wireless headphones"
```

the model converts it into a high-dimensional vector like this (simplified for illustration):

```
[0.12, -0.34, 0.56, ..., 0.78]
```

The exact value are not important. What matters is how close this vector is to others. Text with similar meaning end up near each other in this vector space.

## What Embeddings Represent

Embedding capture semantic meaning rather than grammar or keywords. They encode things like topic, intent, and context.

This is why phrases such as:

- "wireless headphones"
- "bluetooth earphones"
- "cordless audio devices"

end up with similar vectors, even though the words differ. The model recognizes they all relate to the same concept: portable audio equipment.

This behavior is the foundation of semantic search, recommendation systems, retrieval augmented generation, and clustering tasks.

## Vector Size and Tradeoffs

Different embedding models output vectors of varying sizes. Common ranges include 384, 768, and 1536 dimensions.

Smaller vectors are faster and cheaper to store, but they may lose some semantic precision. Larger vectors capture more nuance but increase cost and storage requirements.

In practice, lower dimensions work well for large-scale systems where speed and cost matter. Higher dimensions are useful when accuracy is critical, such as legal or medical search.

## Measuring Similarity

Once text is represented as vectors, similarity becomes a distance calculation.

The most common approaches are cosine similarity, Euclidean distance, and dot product. For most embedding models, **cosine similarity** is the default choice.

When vectors are normalized, cosine similarity and dot product behave very similarly. This is why normalization is often applied before comparison.

## Why Normalization Helps

Raw vector can have different magnitudes. Normalization scales thems so comparisons focus on direction rather than length.

This leads to more stable and consistent similarity scores. In most cases, normalizing embeddings improves reliability and interpretability.

Unless there is a specific reason not to, normalization is a safe default.

## Building a Simple Semantic Search Pipeline

To understand how this works end to end, I built a small embedding explorer instead of relying entirely on high-level libraries.

The tool generates embeddings, stores them locally, computes similarity scores, and performs semantic search. It also supports visualizing embeddings in two dimensions.

Working through the full pipeline makes the mechanics of semantic search much easier to reason about.

## Dataset Choices

To test behavior across different domains, I used a mix of datasets:

- Movie plot summaries
- Product descriptions
- News headlines
- Code snippets in multiple languages

These datasets make clustering patterns and similarity relationships easy to observe.

## Visualizing Vector Space

High-dimensional vectors cannot be visualized directly. Dimensionality reduction techniques like t-SNE and UMAP project them into two dimensions.

When plotted, meaningful patterns appear. Related items cluster together. Translations sit near their source text. Code snippets group by language.

Seeing these plots helps demystify semantic search. The behavior becomes intuitive rather than abstract.

## Testing Search Quality

Queries such as “wireless headphones” returned results like bluetooth earbuds and cordless over-ear headphones.

Even without shared keywords, the intent remained intact. This is where semantic search clearly outperforms keyword matching.

## Handling Synonyms and Translations

Tests with synonyms, paraphrases, and translations showed consistent behavior. Terms like car and automobile clustered together. Translated sentences stayed close to their English versions.

This reinforces a key point: embeddings represent meaning, not surface form.

## Performance Considerations

Embedding generation affects both cost and latency. Batching requests and caching results make a significant difference.

Once embeddings are generated and stored, search becomes fast and inexpensive. Most of the cost is paid upfront during embedding creation.

## Tool Overview

The embedding explorer includes embedding generation, caching, similarity calculations, visualization exports, and performance benchmarks.

It is designed to be model-agnostic so different embedding providers can be tested without changing the core logic.

## Conclusion

Embeddings turn meaning into vectors. Similar meanings result in nearby vectors. Cosine similarity is the most common comparison method. Normalization improves consistency. Visualization helps build intuition. Semantic search consistently outperforms keyword search for real-world queries. Performance depends heavily on batching and caching.
