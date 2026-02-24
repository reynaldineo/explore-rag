# Simple RAG System

A minimal Retrieval-Augmented Generation (RAG) system using Ollama for both embeddings and LLM.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required models pulled: `nomic-embed-text` and `llama3.2:` (or your preferred LLM model)

## Quick Start

1. Ensure Ollama is running: `ollama serve`
2. Pull the models:
    ```
    ollama pull nomic-embed-text
    ollama pull llama2
    ```
3. Place your documents in the `docs/` folder (supports .txt, .md, .html, .csv)
4. Run the RAG system:
    ```
    python simple-rag.py --input docs/
    ```
    Or use the quick run script:
    ```
    ./run.sh
    ```

## Usage Examples

Ingest documents and start interactive mode:
```
python simple-rag.py --input docs/ --chunk-size 500 --top-k 5
```

Save the vector index for faster reloading:
```
python simple-rag.py --input docs/ --save-index index.faiss --metadata-path chunks.json
```

Load a saved index:
```
python simple-rag.py --load-index index.faiss --metadata-path chunks.json --input docs/
```

## Options

- `--input`: Folder containing documents (required)
- `--chunk-size`: Size of text chunks (default: 500)
- `--overlap`: Overlap between chunks (default: 100)
- `--top-k`: Number of top results to retrieve (default: 5)
- `--threshold`: Similarity threshold (default: 0.3)
- `--embedding-model`: Embedding model name (default: nomic-embed-text)
- `--llm-model`: LLM model name (default: llama2)
- `--save-index`: Path to save FAISS index
- `--load-index`: Path to load FAISS index
- `--metadata-path`: Path to metadata JSON

## Interactive Mode

After running, type your questions in the prompt. Type 'exit' or 'quit' to stop.

## Sample Questions for RAG System Testing:

1. What is Retrieval-Augmented Generation (RAG)?
2. How does RAG work?
3. What are the key components of RAG?
4. What are the benefits of using RAG?
5. What are some common use cases for RAG?
6. What challenges does RAG face?
7. Explain the RAG architecture overview.
8. What are the implementation steps for RAG?
9. What tools are popular for building RAG systems?
10. What are the advantages of RAG over pure generation?
11. What are the benefits of RAG according to the HTML document?
12. How does RAG compare to traditional methods?
13. What are some applications of RAG?
14. What is the role of FAISS in RAG?
15. How does BERT compare to Sentence Transformers in RAG?
16. What are the strengths and weaknesses of GPT-3?
17. Why is RAG considered a hybrid approach?
18. What is the use case for Pinecone in RAG?
19. How can I get started with implementing RAG?
20. What are future directions for RAG development?