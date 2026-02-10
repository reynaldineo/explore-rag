# Conversational RAG Quickstart

This guide will help you get started with the Conversational RAG system, which combines retrieval-augmented generation (RAG) with conversational memory for interactive Q&A based on your documents.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Ollama installed and running locally
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd /path/to/learn-rag
   ```

2. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama:**
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull llama3.2
     ```
   - Start Ollama server (it runs on `http://localhost:11434` by default)

## Usage

### Preparing Documents

Place your documents (`.txt` or `.md` files) in a folder. For example, create a `docs/` folder and add your text files there.

### Running the System

Run the conversational RAG script:

```bash
python conversational/conversational_rag.py --docs /path/to/your/docs --session my_session
```

- `--docs`: Path to the folder containing your documents
- `--session`: Optional session ID for conversation persistence (defaults to "default")
- `--top_k`: Optional number of top retrieval results (defaults to 5)

### User Flow

1. **Initialization:**
   - The system loads documents from the specified folder
   - Builds a vector store using Sentence Transformers and FAISS
   - Initializes conversation memory

2. **Interactive Chat:**
   - Enter your questions in the prompt: `You: `
   - The system:
     - Resolves any references (e.g., pronouns) using conversation history
     - Retrieves relevant document chunks
     - Generates a response using the local Ollama LLM
     - Updates conversation memory
   - Type `exit` or `quit` to end the session

3. **Session Persistence:**
   - Conversations are saved to `sessions/{session_id}.json`
   - Previous context is loaded on restart

### Example Interaction

```
You: What is the capital of France?
Assistant: Based on the provided context, the capital of France is Paris.

You: Tell me more about it.
Assistant: Referring to Paris, it is a major city known for landmarks like the Eiffel Tower...
```

## Configuration

- **LLM Model:** Currently set to `llama3.2`. Change `MODEL_NAME` in the code if needed.
- **Embedding Model:** Uses `all-MiniLM-L6-v2` for vector embeddings.
- **Conversation Memory:** Manages up to 6 recent turns, with summarization for longer histories.

## Troubleshooting

- **Ollama not responding:** Ensure Ollama is running and the model is pulled.
- **Import errors:** Check that all dependencies are installed.
- **No documents found:** Verify the `--docs` path contains `.txt` or `.md` files.
- **Slow startup:** Building the vector store may take time for large document sets.

## Advanced Usage

- Modify chunk size and overlap in `VectorStore.add_documents()`
- Adjust temperature and max tokens in `call_llm()` for different response styles
- Extend with custom document loaders or different embedding models</content>
<parameter name="filePath">/home/zord/learn-rag/conversational/QUICKSTART.md