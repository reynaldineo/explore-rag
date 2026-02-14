# 12_agentic_rag.py

import os
import json
import time
import math
import uuid
import logging
import traceback
from typing import Any, Dict, List, Callable, Optional

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

try:
    import openai
except ImportError:
    openai = None  # Optional; system can run without LLM


# ----------------------------
# Logging Setup
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ----------------------------
# Utility Functions
# ----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------------------
# Document Store (Simple RAG Backend)
# ----------------------------

class DocumentStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.index = None
        self.dim = self.model.get_sentence_embedding_dimension()

    def add_documents(self, docs: List[Dict[str, Any]]):
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                self.texts.append(chunk)
                self.metadatas.append({
                    "doc_id": doc.get("id"),
                    "source": doc.get("source"),
                    "chunk_index": i,
                    "title": doc.get("title"),
                })

        embeddings = self.model.encode(self.texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []

        query_emb = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "score": float(score),
            })
        return results


# ----------------------------
# Tool Abstraction
# ----------------------------

class Tool:
    def __init__(self, name: str, description: str, func: Callable[[Any], Any]):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_data: Any) -> Any:
        return self.func(input_data)


# ----------------------------
# Tools Implementation
# ----------------------------

class Tools:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store

    def document_search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query")
        top_k = input_data.get("top_k", 5)
        results = self.doc_store.search(query, top_k=top_k)
        return {"results": results}

    def web_search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Mock web search. Replace with real API if desired.
        query = input_data.get("query")
        logging.info(f"Mock web search for: {query}")

        mock_results = [
            {"title": "Industry Revenue Growth Report 2024", "content": "Average industry revenue growth in 2024 was 12%."},
            {"title": "Market Trends Overview", "content": "The market average growth hovered between 10% and 13% last year."},
        ]
        return {"results": mock_results}

    def calculator(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        expression = input_data.get("expression")
        try:
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def python_repl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code = input_data.get("code")
        local_vars = {}
        try:
            exec(code, {}, local_vars)
            return {"output": local_vars}
        except Exception as e:
            return {"error": str(e)}


# ----------------------------
# Agent Core
# ----------------------------

class Agent:
    def __init__(
        self,
        tools: Dict[str, Tool],
        llm_model: Optional[str] = None,
        max_steps: int = 8,
        temperature: float = 0.2,
    ):
        self.tools = tools
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.temperature = temperature
        self.trace: List[Dict[str, Any]] = []

    def run(self, question: str) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        state = {
            "question": question,
            "history": [],
            "final_answer": None,
        }

        logging.info(f"Session {session_id} started.")
        for step in range(self.max_steps):
            thought = self._reason(state)
            action = self._decide_action(thought)
            observation = self._act(action)

            state["history"].append({
                "thought": thought,
                "action": action,
                "observation": observation,
            })
            self.trace.append(state["history"][-1])

            if self._is_final(observation):
                state["final_answer"] = observation.get("final_answer")
                logging.info(f"Session {session_id} completed in {step+1} steps.")
                break

        if not state.get("final_answer"):
            state["final_answer"] = "Unable to complete the task within the step limit."

        return state

    # ----------------------------
    # Reasoning, Acting, Observing
    # ----------------------------

    def _reason(self, state: Dict[str, Any]) -> str:
        """
        Generate reasoning text based on state.
        If an LLM is available, use it. Otherwise, use a heuristic planner.
        """
        if openai and self.llm_model:
            return self._reason_with_llm(state)
        else:
            return self._heuristic_reason(state)

    def _reason_with_llm(self, state: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning agent using tools to answer questions."},
            {"role": "user", "content": f"Question: {state['question']}"},
        ]
        for turn in state["history"]:
            messages.append({"role": "assistant", "content": f"Thought: {turn['thought']}"})
            messages.append({"role": "assistant", "content": f"Action: {json.dumps(turn['action'])}"})
            messages.append({"role": "assistant", "content": f"Observation: {json.dumps(turn['observation'])}"})

        messages.append({"role": "user", "content": "What should you do next? Respond with your thought."})

        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _heuristic_reason(self, state: Dict[str, Any]) -> str:
        """
        Simple planner based on keyword heuristics and conversation state.
        """
        question = state["question"].lower()
        history = state["history"]

        if not history:
            if "compare" in question and ("industry" in question or "market" in question):
                return "I need internal data first, then industry benchmarks."
            if "calculate" in question or any(op in question for op in ["+", "-", "*", "/"]):
                return "This looks like a calculation task."
            return "I should retrieve relevant internal documents."

        last_obs = history[-1]["observation"]

        if "error" in last_obs:
            return "The last tool failed. I should try a different approach or rephrase the query."

        if "results" in last_obs and isinstance(last_obs["results"], list):
            # Decide next step based on what was retrieved
            if any("industry" in str(r).lower() for r in last_obs["results"]):
                return "I now have industry data. I should compare and answer."
            if any("revenue" in str(r).lower() for r in last_obs["results"]):
                return "I have internal revenue data. I should look up industry averages."
            return "I have some data. I should synthesize an answer."

        if "result" in last_obs:
            return "I have the calculation result. I should answer."

        return "I should try to answer now."

    def _decide_action(self, thought: str) -> Dict[str, Any]:
        thought_lower = thought.lower()

        if "internal" in thought_lower or "document" in thought_lower:
            return {"tool": "document_search", "input": {"query": self._extract_query(thought), "top_k": 5}}

        if "industry" in thought_lower or "market" in thought_lower or "web" in thought_lower:
            return {"tool": "web_search", "input": {"query": self._extract_query(thought)}}

        if "calculate" in thought_lower or any(op in thought_lower for op in ["+", "-", "*", "/"]):
            return {"tool": "calculator", "input": {"expression": self._extract_expression(thought)}}

        if "python" in thought_lower or "analyze" in thought_lower:
            return {"tool": "python_repl", "input": {"code": self._extract_code(thought)}}

        # Default: attempt to answer directly
        return {"tool": "final_answer", "input": {"text": "Answer based on available information."}}

    def _act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = action.get("tool")
        tool_input = action.get("input", {})

        if tool_name == "final_answer":
            return {"final_answer": tool_input.get("text")}

        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool.run(tool_input)
        except Exception as e:
            logging.error(traceback.format_exc())
            return {"error": str(e)}

    def _is_final(self, observation: Dict[str, Any]) -> bool:
        return "final_answer" in observation

    # ----------------------------
    # Helper Parsers
    # ----------------------------

    def _extract_query(self, text: str) -> str:
        return text.replace("I need", "").replace("I should", "").strip()

    def _extract_expression(self, text: str) -> str:
        # Very simple placeholder
        return text.split(":")[-1].strip()

    def _extract_code(self, text: str) -> str:
        # Placeholder: user can improve this
        return "print('No code provided')"


# ----------------------------
# Example Usage
# ----------------------------

def load_example_documents() -> List[Dict[str, Any]]:
    return [
        {
            "id": "doc1",
            "title": "Company Revenue Report 2024",
            "source": "internal_finance",
            "text": "Our company revenue grew by 18% in 2024, driven by strong performance in the cloud services division.",
        },
        {
            "id": "doc2",
            "title": "Product Performance Summary",
            "source": "internal_product",
            "text": "Product A experienced 18% growth, Product B grew by 10%, and Product C grew by 7% last year.",
        },
        {
            "id": "doc3",
            "title": "Market Overview",
            "source": "internal_strategy",
            "text": "The overall market environment in 2024 was stable, with moderate growth across most sectors.",
        },
    ]


def build_agent() -> Agent:
    doc_store = DocumentStore()
    doc_store.add_documents(load_example_documents())

    tools_impl = Tools(doc_store)

    tools = {
        "document_search": Tool(
            name="document_search",
            description="Search internal documents using vector similarity.",
            func=tools_impl.document_search,
        ),
        "web_search": Tool(
            name="web_search",
            description="Search the web for current information.",
            func=tools_impl.web_search,
        ),
        "calculator": Tool(
            name="calculator",
            description="Evaluate a mathematical expression.",
            func=tools_impl.calculator,
        ),
        "python_repl": Tool(
            name="python_repl",
            description="Run Python code for data analysis.",
            func=tools_impl.python_repl,
        ),
    }

    agent = Agent(
        tools=tools,
        llm_model=None,  # Set to model name if using OpenAI
        max_steps=8,
        temperature=0.2,
    )

    return agent


def run_cli():
    agent = build_agent()
    print("Agentic RAG CLI. Type 'exit' to quit.\n")

    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = agent.run(user_input)

        print("\n--- Final Answer ---")
        print(result["final_answer"])

        print("\n--- Trace ---")
        for i, step in enumerate(result["history"], 1):
            print(f"\nStep {i}")
            print("Thought:", step["thought"])
            print("Action:", step["action"])
            print("Observation:", step["observation"])


if __name__ == "__main__":
    run_cli()
