# RAG From Scratch — Detailed Notes

This document provides a detailed explanation of the concepts implemented in the
`Basic_Fundamental_rag_from_scratch_1_to_4.ipynb` notebook. It is intended as a
reference to revisit the ideas later and reinforce understanding.

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is a hybrid NLP approach combining
information retrieval with generative models. Instead of relying solely on the
knowledge encoded in a language model, RAG systems retrieve real documents or
text chunks from an external corpus and supply them as context to the model.
This enables up-to-date, accurate responses and reduces hallucinations.

### Key components:

1. **Document Corpus**: The collection of text sources (web pages, PDFs, notes,
etc.) that we want the system to "know" about.
2. **Indexer**: Processes documents, splits them into manageable chunks,
   generates vector embeddings, and stores them in a vector database (a
   "vectorstore").
3. **Retriever**: Given a query, computes an embedding and searches the vector
   database for the most semantically similar chunks.
4. **Generator (LLM)**: Takes the retrieved context and the user's question,
   constructs a prompt, and generates an answer.

---

## 2. Notebook Structure & Purpose

The notebook is organized to progressively introduce each piece of the RAG
pipeline, with both conceptual notes and runnable code samples.

### Environment Setup & API Keys

- Installs required packages (`langchain`, `chromadb`, `langchain_community`,
  `ollama`, etc.).
- Configures environment variables for LangSmith tracing and API keys
  (OpenAI, LangSmith, etc.).

These cells ensure reproducibility and allow you to plug in your own keys or
local models.

### Part 1: Overview

This section demonstrates an end-to-end RAG flow using a single code block.
It:

1. Loads a web document with `WebBaseLoader`.
2. Splits it into chunks with `RecursiveCharacterTextSplitter`.
3. Embeds chunks using `OllamaEmbeddings`.
4. Builds a Chroma vectorstore and creates a retriever.
5. Defines a prompt pulled from LangSmith’s prompt hub.
6. Constructs a chain that retrieves context, formats it, and feeds it to an
   Ollama LLM.
7. Invokes the chain with a sample question.

This quick example is valuable as a template for later experiments.

### Part 2: Indexing Deep Dive

Explores each indexing step with explanatory comments and small experiments:

- **Token Counting** – uses `tiktoken` to show how many tokens a string uses.
- **Embedding Models** – compares OpenAI with local models like Nomic via
  `OllamaEmbeddings`.
- **Calculating Cosine Similarity** – demonstrates how to measure embedding
  similarity manually.
- **Loaders** – shows how to fetch a blog post with parsing settings.
- **Text Splitting** – uses a `RecursiveCharacterTextSplitter` (with
token-based chunking) to divide a document into semantic pieces.
- **Vectorstore Construction** – builds a `Chroma` store from the splits and
  creates a retriever.

These cells help you understand what goes into building the index and how the
parameters affect chunk size, overlap, and embedding length.

### Part 3: Retrieval

Illustrates how to query the retriever directly using `invoke` or
`get_relevant_documents`. It also shows adjusting `search_kwargs` (e.g. `k`)
to control how many chunks are returned.

### Part 4: Generation

Demonstrates building a prompt template, initializing an LLM (ChatOpenAI or
Ollama), and composing a chain to answer questions. The section also includes
examples of using the LangSmith prompt hub and different prompt/LLM
combinations.

Later examples expand on this with RAG-specific chain syntax that pipes the
retriever output through formatting functions before passing it to the model.

---

## 3. Reusing the Notebook

When returning to the notebook instead of rereading all the cells, keep this
file bookmarked. 

- **Refer to the overview** to recall the full pipeline quickly.
- **Scan the indexing section** when you need to rebuild or modify your data
  preprocessing steps.
- **Use the retrieval/generation examples** as templates for new queries or
  alternative LLMs.

---

## 4. Tips & Extensions

- You can swap out `OllamaEmbeddings` or `OllamaLLM` for OpenAI, HuggingFace, or
  other providers by changing a few lines.
- The notebook uses Chroma vectorstore, but LangChain supports others like
  FAISS, Supabase, Pinecone, etc. Experiment to match your deployment needs.
- LangSmith tracing is enabled; check the dashboard to inspect run metadata,
  prompts, and responses.

---

Keep this document updated as you experiment further, and add links to new
resources or advanced techniques you discover.