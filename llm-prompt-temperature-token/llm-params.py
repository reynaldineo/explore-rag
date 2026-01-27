import os
import json
import time
import random
import re
import requests
from datetime import datetime
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "llama3.2" 
OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_FILE = "llm-prompt-temperature-token_results.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

TEMPERATURES = [0, 0.5, 1.0, 1.5]
RUNS_PER_EXPERIMENT = 2

# =========================
# UTILITY FUNCTIONS
# =========================

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def log_json(filename: str, data: List[Dict[str, Any]]):
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_json(text: str) -> Dict[str, Any]:
    """
    Attempts to extract JSON object from a model response.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"error": "Failed to parse JSON", "raw_output": text}

# =========================
# API CALL WRAPPER
# =========================

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
    print(f"Calling LLM with temperature={temperature}, max_tokens={max_tokens}")
    start_time = time.time()
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    latency = time.time() - start_time
    print(f"LLM call completed in {latency:.2f} seconds")

    # https://docs.ollama.com/api/chat
    content = result["message"]["content"]
    prompt_tokens = result.get("prompt_eval_count")
    completion_tokens = result.get("eval_count")
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency": latency
    }

# =========================
# PROMPT PATTERNS
# =========================

def system_vs_user_prompt(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]   

def single_user_prompt(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": user_prompt}
    ]

def few_shot_prompt(examples: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    for ex in examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def chain_of_thought_prompt(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Show your reasoning step by step."},
        {"role": "user", "content": user_prompt}
    ]

def json_mode_prompt(user_prompt: str, with_examples: bool = False) -> List[Dict[str, str]]:
    messages = []
    if with_examples:
        examples = [
            {
                "user": "Provide the following information in JSON format: name, age, city. My name is Alice, I am 30 years old and I live in New York.",
                "assistant": json.dumps({"name": "Alice", "age": 30, "city": "New York"}),
            },
            {
                "user": "Provide the following information in JSON format: name, age, city. My name is Bob, I am 25 years old and I live in Los Angeles.",
                "assistant": json.dumps({"name": "Bob", "age": 25, "city": "Los Angeles"}),
            }
        ]
        return few_shot_prompt(examples, user_prompt)
    else:
        messages.append({"role": "system", "content": "Respond ONLY in valid JSON. No explanation."})
        messages.append({"role": "user", "content": user_prompt})
        return messages

# =========================
# EXPERIMENT RUNNERS
# =========================

def run_temperature_experiment(prompt: str, label: str) -> List[Dict[str, Any]]:
    results = []
    for temp in TEMPERATURES:
        for run in range(RUNS_PER_EXPERIMENT):
            messages = single_user_prompt(prompt)
            result = call_llm(messages, temperature=temp)
            record = build_result_record(
                experiment_type="temperature_test",
                label=label,
                temperature=temp,
                run_id=run,
                prompt=prompt,
                response=result,
            )
            results.append(record)
    return results

def run_json_extraction_experiment(prompts: List[str]) -> List[Dict[str, Any]]:
    results = []
    for prompt in prompts:
        for with_examples in [False, True]:
            for run in range(RUNS_PER_EXPERIMENT):
                messages = json_mode_prompt(prompt, with_examples=with_examples)
                result = call_llm(messages, temperature=0.3)
                extracted_json = extract_json(result["content"])
                record = build_result_record(
                    experiment_type="json_extraction",
                    label="with_examples" if with_examples else "without_examples",
                    temperature=0.3,
                    run_id=run,
                    prompt=prompt,
                    response=result,
                    extracted_json=extracted_json,
                )
                results.append(record)
    return results

def run_chain_of_thought_experiment(prompts: List[str]) -> List[Dict[str, Any]]:
    results = []
    for prompt in prompts:
        for run in range(RUNS_PER_EXPERIMENT):
            messages = chain_of_thought_prompt(prompt)
            result = call_llm(messages, temperature=0.7)
            record = build_result_record(
                experiment_type="chain_of_thought",
                label="math_reasoning",
                temperature=0.7,
                run_id=run,
                prompt=prompt,
                response=result,
            )
            results.append(record)
    return results


# =========================
# RESULT STRUCTURE
# =========================

def build_result_record(
    experiment_type: str,
    label: str,
    temperature: float,
    run_id: int,
    prompt: str,
    response: Dict[str, Any],
    extracted_json: Dict[str, Any] = None,
) -> Dict[str, Any]:
    record = {
        "timestamp": now_iso(),
        "experiment_type": experiment_type,
        "label": label,
        "temperature": temperature,
        "run_id": run_id,
        "model": MODEL_NAME,
        "prompt": prompt,
        "output": response["content"],
        "tokens": {
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "total_tokens": response["total_tokens"],
        },
        "latency_seconds": round(response["latency"], 3),
    }

    if extracted_json is not None:
        record["extracted_json"] = extracted_json

    return record

# =========================
# EDGE CASE TESTS
# =========================

def run_edge_case_tests() -> List[Dict[str, Any]]:
    results = []
    edge_case_prompts = [
        "",  
        "     ",  
        "What is the meaning of life?" * 100,  
        "Tell me a joke." + "😂" * 50,  
        "Translate to Indonesia: Hello, how are you?"  
    ]

    for prompt in edge_case_prompts:
        for run in range(RUNS_PER_EXPERIMENT):
            messages = single_user_prompt(prompt)
            result = call_llm(messages, temperature=0.5)
            record = build_result_record(
                experiment_type="edge_case_test",
                label="varied_inputs",
                temperature=0.5,
                run_id=run,
                prompt=prompt,
                response=result,
            )
            results.append(record)
    return results

# =========================
# MAIN EXPERIMENT EXECUTION
# =========================

def main():
    all_results = []

    # 1️. Temperature Experiment
    temp_prompt = "Write a short story about a robot learning to love."
    all_results.extend(run_temperature_experiment(temp_prompt, label="story_generation"))

    # 2. JSON Extraction Experiment
    json_prompts = [
        "Provide the following information in JSON format: name, age, city. My name is Charlie, I am 28 years old and I live in Chicago.",
        "Provide the following information in JSON format: name, age, city. My name is Dana, I am 35 years old and I live in Miami."
    ]
    all_results.extend(run_json_extraction_experiment(json_prompts))

    # 3️. Chain of Thought Experiment
    cot_prompts = [
        "If a train travels 60 miles in 1 hour and 30 minutes, what is its average speed in miles per hour?",
        "A store had 150 apples. If they sold 45 apples and then received a shipment of 30 more, how many apples do they have now?"
    ]
    all_results.extend(run_chain_of_thought_experiment(cot_prompts))

    # 4️. Edge Case Testing
    all_results.extend(run_edge_case_tests())

    # Save all results as JSON array
    log_json(OUTPUT_FILE, all_results)

    print(f"✅ Experiments complete! Results saved to {os.path.join(LOG_DIR, OUTPUT_FILE)}")
    print(f"📁 Detailed logs stored in {LOG_DIR}/")


if __name__ == "__main__":
    main()