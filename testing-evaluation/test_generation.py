import sys
sys.path.append('/home/zord/learn-rag/testing-evaluation')

from app import ExampleRAGSystem

# Initialize the RAG system
rag = ExampleRAGSystem(config={"k": 2, "docs_folder": "../docs"})

# Test generation for the first question
question1 = "What is enterprise billing?"
retrieval = rag.retrieve(question1)
generation = rag.generate(question1, retrieval.retrieved_chunks)
print("Question:", question1)
print("Retrieved chunks IDs:", retrieval.retrieved_chunks)
print("Generated answer:", generation.answer)
print("Expected:", "Enterprise billing refers to the complex process of invoicing and payment collection for large organizations.")
print()

# Test for second question
question2 = "What are the key principles of enterprise billing?"
retrieval2 = rag.retrieve(question2)
generation2 = rag.generate(question2, retrieval2.retrieved_chunks)
print("Question:", question2)
print("Retrieved chunks IDs:", retrieval2.retrieved_chunks)
print("Generated answer:", generation2.answer)
print("Expected:", "Accuracy, Scalability, Flexibility, Compliance, Automation")
print()

# Print the context for second question
print("Context for second question:")
context_parts = []
for chunk_id_str in retrieval2.retrieved_chunks:
    chunk_id = int(chunk_id_str)
    chunk = rag.vector_store.chunks[chunk_id]
    context_parts.append(chunk.text)
context = "\n\n".join(context_parts)
print(context[:1000])  # first 1000 chars