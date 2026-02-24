# RAG API

A production-ready REST API for Retrieval-Augmented Generation (RAG) using local Ollama for embeddings and LLM.

## Installation

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn python-dotenv pydantic starlette requests numpy faiss-cpu
   ```

2. Install and start Ollama:
   - Download from [ollama.ai](https://ollama.ai)
   - Pull models: `ollama pull nomic-embed-text` and `ollama pull llama3.2`

## Starting the API

1. Set environment variables (optional):
   - `API_KEY`: API key for authentication (default: "dev-secret-key")
   - `RATE_LIMIT`: Requests per minute (default: 60)

2. Run the server:
   ```bash
   uvicorn app:app --reload
   ```

The API will be available at `http://localhost:8000`.

## Usage

- **Health Check**: GET `/health`
- **Query**: POST `/query` with JSON `{"question": "Your question"}`
- **Add Document**: POST `/documents` with JSON `{"document_id": "id", "text": "content"}`
- **Delete Document**: DELETE `/documents/{document_id}`

Include `X-API-Key` header for authentication.

## Explanation

The API ingests documents, chunks them, embeds using Ollama, stores in FAISS vector DB, and answers queries by retrieving relevant chunks and generating responses via Ollama LLM.</content>
<parameter name="filePath">/home/zord/learn-rag/rag-api/README.md