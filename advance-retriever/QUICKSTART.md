# Quick Start

## Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Ingest Documents
1. Place `.txt` or `.md` files in a folder (e.g., `docs/`).
2. Run: `python advanced_retrieval.py --docs docs/`

## Retrieve
Run: `python advanced_retrieval.py --docs docs/ --query "your search query" --top_k 5 --output results.json`

Options:
- `--no_hybrid`: Dense search only
- `--no_query_expansion`: Skip expansion
- `--no_reranking`: Skip reranking
- `--no_mmr`: Skip MMR
- `--no_parent_retrieval`: Return chunks instead of full docs

## Evaluate
1. Create `eval.json`: `[{"query": "example", "relevant": ["doc1", "doc2"]}]`
2. Run: `python advanced_retrieval.py --docs docs/ --evaluate eval.json --top_k 5`

Results saved to `evaluation_results.json`.


---

Summary Example:
```
python advanced_retrieval.py \
  --docs ./../docs \
  --query "How does enterprise billing work?" \
  --top_k 5 \
  --no_query_expansion \
  --output results.json

python advanced_retrieval.py --docs ./../docs --evaluate eval_queries.json
```