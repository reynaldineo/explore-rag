# Embedding Explorer

A Python script for exploring text embeddings using Ollama's local embedding models. This tool demonstrates semantic search, embedding visualization, similarity analysis, and performance benchmarking.

## What It Does

This script provides a comprehensive exploration of text embeddings:

- **Text Embedding Generation**: Converts text into high-dimensional vector representations using Ollama's embedding models
- **Semantic Search**: Finds the most relevant texts to a query based on vector similarity
- **Similarity Analysis**: Builds and analyzes similarity matrices between all text pairs
- **Visualization**: Creates 2D t-SNE plots to visualize embedding relationships
- **Performance Benchmarking**: Measures embedding generation speed and token usage

## Prerequisites

- Python 3.7+
- Ollama installed and running locally
- Required Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - matplotlib
  - scikit-learn
  - requests

## Quick Start

1. **Start Ollama**: Ensure Ollama is running on `http://localhost:11434`
   ```bash
   ollama serve
   ```

2. **Pull the embedding model**:
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy matplotlib scikit-learn requests
   ```

4. **Run the script**:
   ```bash
   python embedding.py
   ```

## Code Flow

The main execution flow follows these steps:

1. **Embedding Generation**: Processes the built-in dataset of 25 diverse texts (movies, products, news, code, misc) and generates embeddings for each

2. **Similarity Matrix Construction**: Creates a cosine similarity matrix showing relationships between all text pairs

3. **Visualization**: Uses t-SNE to reduce embeddings to 2D and creates a scatter plot with text labels

4. **Semantic Search**: Tests 5 sample queries against the dataset, returning top 5 most similar results

5. **Benchmarking**: Measures total processing time, average time per text, and token usage statistics

## Outputs

The script generates several output files in the `embedding_outputs/` directory:

- `similarity_matrix.json`: Cosine similarity scores between all text pairs
- `embedding_tsne_plot.png`: 2D visualization of embeddings using t-SNE
- `semantic_search_results.json`: Search results for test queries
- `embedding_benchmark.json`: Performance metrics and statistics

## Configuration

Key settings can be modified at the top of the script:

- `MODEL_NAME`: Embedding model to use (default: "nomic-embed-text")
- `OLLAMA_URL`: Ollama API endpoint (default: "http://localhost:11434/api/embed")
- `CACHE_DIR`: Directory for caching embeddings (default: "embedding_cache")
- `OUTPUT_DIR`: Directory for output files (default: "embedding_outputs")

## Caching

Embeddings are automatically cached to avoid redundant API calls. Cached data includes the embedding vector and metadata (model, tokens, latency).

## Dataset

The script uses a built-in dataset of 25 sample texts covering various domains:
- 5 movie plot summaries
- 5 product descriptions
- 5 news headlines
- 5 code snippets
- 5 miscellaneous topics

## Customization

To use your own data:
1. Replace the `DATASET` list with your texts
2. Modify `test_queries` in the main function for custom search tests
3. Adjust similarity metrics (cosine, euclidean, dot product) as needed

## Model Comparison

The script includes a hook for comparing multiple embedding models. Uncomment the model comparison section in `main()` and specify models to test.

## Troubleshooting

- **Connection Error**: Ensure Ollama is running and accessible at the configured URL
- **Model Not Found**: Pull the required model using `ollama pull <model_name>`
- **Import Errors**: Install missing dependencies with `pip install`
- **Visualization Issues**: Ensure matplotlib backend supports PNG output