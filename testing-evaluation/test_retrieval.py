import sys
sys.path.append('/home/zord/learn-rag/testing-evaluation')

from app import ExampleRAGSystem

# Initialize the RAG system
rag = ExampleRAGSystem(config={"k": 5, "docs_folder": "../docs"})

# Test retrieval for the first question
question1 = "What is enterprise billing?"
retrieval = rag.retrieve(question1)
print("Question:", question1)
print("Retrieved chunks (first 200 chars each):")
for i, chunk in enumerate(retrieval.retrieved_chunks):
    print(f"Chunk {i}: {chunk[:200]}...")
print()

# Test for second question
question2 = "What are the key principles of enterprise billing?"
retrieval2 = rag.retrieve(question2)
print("Question:", question2)
print("Retrieved chunks (first 200 chars each):")
for i, chunk in enumerate(retrieval2.retrieved_chunks):
    print(f"Chunk {i}: {chunk[:200]}...")
print()

# Also print chunk IDs
print("Chunk IDs for question 1:")
for idx, score in rag.vector_store.search(rag.embedding_model.embed_texts([question1])[0], 5):
    if idx >= 0:
        chunk = rag.vector_store.chunks[idx]
        print(f"ID: {chunk.id}, Score: {score:.3f}, Source: {chunk.source}")
print()

print("Chunk IDs for question 2:")
for idx, score in rag.vector_store.search(rag.embedding_model.embed_texts([question2])[0], 5):
    if idx >= 0:
        chunk = rag.vector_store.chunks[idx]
        print(f"ID: {chunk.id}, Score: {score:.3f}, Source: {chunk.source}")