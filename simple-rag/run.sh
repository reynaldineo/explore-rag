#!/bin/bash

# Quick run script for Simple RAG System

echo "Starting Simple RAG System..."
echo "Make sure Ollama is running and models are pulled."
echo "Usage: ./run.sh [input_folder] [options]"
echo ""

INPUT_DIR=${1:-docs/}
shift

python simple-rag.py --input "$INPUT_DIR" "$@"