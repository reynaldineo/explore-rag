# Multimodal RAG System (multimodal.py)

## Overview

This is a Retrieval-Augmented Generation (RAG) system designed to handle multimodal content, including both text and images extracted from documents such as PDFs. It leverages CLIP (Contrastive Language-Image Pretraining) for generating embeddings of text and images, stores them in an in-memory vector store, and uses a local vision-language model via Ollama (`qwen3-vl:8b`) for dynamic image captioning during ingestion and multimodal answer generation during querying.

The system enables users to ingest documents, retrieve relevant multimodal content based on natural language queries, and generate context-aware answers that incorporate both textual and visual information.

## Features

- **Multimodal Ingestion**: Processes text documents and PDFs, extracting text chunks and images.
- **Image Processing**: Extracts images from PDFs, performs OCR (Optical Character Recognition) using Tesseract, and generates detailed captions using the `qwen3-vl:8b` vision model.
- **Embedding**: Uses CLIP to create unified embeddings for text and images, enabling cross-modal retrieval.
- **Vector Search**: Stores chunks in a simple in-memory vector store and performs cosine similarity search.
- **Answer Generation**: Combines retrieved text and images in a single prompt to the vision LLM for generating coherent, multimodal answers.
- **Extensible**: Built with dataclasses and modular classes for easy extension (e.g., add more embedding models or vector stores).

## Architecture Flow

1. **Ingestion**:
   - For text: Split into chunks, embed with CLIP, store as `TextChunk`.
   - For PDFs: Extract images, run OCR, generate captions via Ollama, embed images with CLIP, store as `ImageChunk`.
   - All chunks are added to the `MultiModalVectorStore`.

2. **Querying**:
   - Embed the query text with CLIP.
   - Search the vector store for top-K similar chunks (text and image).
   - Format results: Include text content, image captions/OCR, and base64-encoded images.
   - Generate answer: Send question, retrieved texts, and images to `qwen3-vl:8b` via Ollama API for multimodal response.

3. **Output**: Returns a dictionary with the question, generated answer, and detailed results (including image data for visualization).

## High-Level Idea (Mental Model)

This system lets you:

1. **Ingest Documents**
    - Raw text files
    - PDFs → extract images → OCR + generate captions

2. **Embed Everything**
    - Text and images into the same vector space using CLIP

3. **Store Embeddings**
    - In-memory vector store for fast retrieval

4. **Query in Natural Language**
    - Retrieve the most relevant text chunks and images
    - Generate answers using retrieved context with the vision LLM

## Setup and Installation

### Prerequisites

- **Python**: 3.8 or higher.
- **Ollama**: Install from [ollama.ai](https://ollama.ai/download). Ensure it's running locally on port 11434.
  - Pull the required model: `ollama pull qwen3-vl:8b` (this downloads ~5-6GB; ensure sufficient disk space and optionally GPU for faster inference).
- **Tesseract OCR**: Install via `sudo apt install tesseract-ocr` (Linux) or equivalent for your OS.
- **PyTorch and CUDA** (optional but recommended for CLIP): If using GPU, ensure CUDA-compatible PyTorch is installed.

### Dependencies

Install Python packages from the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `numpy`, `torch`, `clip-by-openai` (for CLIP embeddings)
- `Pillow` (PIL for image processing)
- `PyMuPDF` (fitz for PDF parsing)
- `pytesseract` (for OCR)
- `scikit-learn` (for cosine similarity)
- `requests` (for Ollama API calls)

### Running Ollama

Start the Ollama server in a separate terminal:

```bash
ollama serve
```

Verify it's running: `curl http://localhost:11434/api/tags` should return available models.

## Quick Start

1. **Activate your virtual environment** (if using one):
   ```bash
   source venv/bin/activate  # Adjust path as needed
   ```

2. **Prepare a sample PDF**: Place a PDF with images in the `multimodal/` directory (e.g., `sample.pdf`). The system will extract and process images.

3. **Run the example**:
   ```bash
   cd multimodal
   python3 multimodal.py
   ```

   This will:
   - Ingest a sample document (text + PDF).
   - Query: "Show me the architecture diagram".
   - Print the JSON result, including the generated answer and retrieved content.

4. **Customize**:
   - Edit `main()` in `multimodal.py` to change the PDF path or query.
   - For production, integrate into a larger app (e.g., via API).

## Usage Examples

### Programmatic Usage

```python
from multimodal import MultiModalRAG

# Initialize
rag = MultiModalRAG()

# Ingest documents
rag.ingest_document(
    document_id="my_doc",
    text="This is a document about AI architectures.",
    pdf_path="path/to/diagram.pdf"  # Optional
)

# Query
result = rag.query("Explain the architecture", top_k=5)
print(result["answer"])  # Multimodal answer from Ollama
```

### API Details

- **Ingestion**: `ingest_document(document_id, text=None, pdf_path=None)`
- **Query**: `query(question, top_k=5)` returns `{"question": str, "answer": str, "results": list}`
- **Ollama Integration**: Uses `/api/chat` endpoint. Handles timeouts and errors gracefully.

## Configuration

- **Embedding Dimensions**: Set via `EMBEDDING_DIM = 512` (CLIP default).
- **Top-K Retrieval**: Default `TOP_K = 5`.
- **Data Directory**: Images saved to `DATA_DIR = "data_multimodal"`.
- **Ollama Model**: Hardcoded to `"qwen3-vl:8b"`; change in `call_ollama_chat` if needed.
- **Logging**: Uses Python's `logging` at INFO level.

## Troubleshooting

- **Ollama Errors**: Ensure `ollama serve` is running and model is pulled. Check logs for API failures.
- **Import Errors**: Install missing packages (e.g., `pip install clip-by-openai pymupdf pytesseract`).
- **No Images in PDF**: System handles text-only documents gracefully.
- **Slow Inference**: Use GPU for CLIP/Ollama; reduce `top_k` for faster queries.
- **Memory Issues**: For large PDFs, process in batches or use a persistent vector store (e.g., FAISS on disk).

## Future Enhancements

- Add support for more document types (e.g., images, videos).
- Integrate persistent storage (e.g., FAISS index on disk).
- Add evaluation metrics for retrieval/answer quality.
- Support multiple vision models or fine-tuning.

For issues or contributions, refer to the main repository.