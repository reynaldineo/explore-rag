# RAG Evaluation Framework

This framework evaluates Retrieval-Augmented Generation (RAG) systems using a golden dataset of questions and expected answers.

## Code Flow

1. **Load Golden Dataset**: Reads a JSON file containing questions, expected answers, relevant chunks, and metadata.

2. **Initialize RAG System**: Creates an instance of the RAG system with configuration parameters. The system ingests documents from the specified folder, chunks them, embeds them using Ollama's nomic-embed-text model, and stores them in a FAISS vector index.

3. **Evaluate Each Example**:
   - Retrieve relevant chunks for the question using cosine similarity search.
   - Generate an answer using the retrieved chunks as context via Ollama's llama3.2 model.
   - Compute retrieval metrics (precision, recall, MRR, NDCG).
   - Compute generation metrics (exact match, ROUGE-L, BLEU, semantic similarity).
   - Determine if the example passes based on semantic similarity threshold (>=0.7).

4. **Generate Reports**: Saves evaluation results to JSON and CSV files, including summary metrics and failure analysis.

5. **Optional Features**:
   - Regression testing against baseline metrics.
   - Parameter sweep to test different configurations.

## Dependencies

- Python 3.8+
- scikit-learn
- nltk
- faiss-cpu
- numpy
- tqdm
- requests
- Ollama (running locally with models: nomic-embed-text, llama3.2)

Install with:

```bash
pip install scikit-learn nltk faiss-cpu numpy tqdm requests
```

Also, download NLTK data:

```python
python -c "import nltk; nltk.download('punkt')"
```

## Setup

1. Ensure Ollama is running locally.

2. Pull the required models:

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. Place documents in the `../docs` folder (or specify via config).

4. Prepare a golden dataset JSON file, e.g.:

   ```json
   [
     {
       "question": "What is the capital of France?",
       "expected_answer": "Paris",
       "relevant_chunks": ["Paris is the capital of France."],
       "difficulty": "easy"
     }
   ]
   ```

## How to Run

```bash
python app.py --dataset golden_dataset.json --output_dir evaluation_reports --k 5
```

Options:
- `--dataset`: Path to golden dataset JSON file (required)
- `--output_dir`: Directory to save reports (default: evaluation_reports)
- `--k`: Top-K retrieval cutoff (default: 5)
- `--baseline`: Path to baseline metrics JSON for regression testing
- `--run_sweep`: Run parameter sweep

## Output

- JSON report with summary metrics, failure analysis, and detailed results.
- CSV report with per-question metrics.
- Optional parameter sweep results.

## Architecture

- **DocumentLoader**: Loads documents from a folder (supports .txt, .md, .html, .csv).
- **Chunker**: Splits documents into overlapping chunks.
- **EmbeddingModel**: Uses Ollama's nomic-embed-text for embeddings.
- **VectorStore**: FAISS index for efficient similarity search.
- **RAGSystem**: Integrates retrieval and generation.
- **Evaluator**: Computes metrics for each example.
- **ReportGenerator**: Aggregates results into reports.