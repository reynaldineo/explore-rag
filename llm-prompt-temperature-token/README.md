# Exploring LLM Behavior with Prompt, Temperature, and Tokens

> Explore how LLMs respond to small changes in prompts, temperature, and tokens, uncovering patterns for consistent results.

Large language models are **powerful**, but their behavior is **sensitive to small changes**. A slight adjustment to a prompt can lead to a very different response. Changing temperature affects creativity and consistency. Increasing token limits increases cost and latency.

Working effectively with LLMs means **understanding how these factors interact**. This post explores how models behave across different prompts, temperature settings, and token limits, based on hands-on testing.

The focus is on:

- How **prompt phrasing** and structure affect reliability
- How **temperature** changes creativity and consistency
- How to get structured data like **JSON** more reliably
- How **costs** and **speed** change with different settings
- How to set up a simple **experiment setup** in Python

## Understanding the LLM Workflow

All LLMs work in basically the same way. Whether it's OpenAI, Anthropic, Ollama, or another provider, the workflow is nearly identical:

1. **Define your prompt:** the text that tells the model what you want.
2. **Set parameters:** things like temperature, max tokens, or stop sequences that influence output.
3. **Send the request:** an API call delivers your prompt and parameters to the model.
4. **Receive the response:** the model returns text that attempts to satisfy your instructions.

Understanding this makes prompt engineering didn't feel like magic and more like cause and effect. Small changes at any step can affect the final output, sometimes slightly, sometimes significantly.

## The Parameters That Shape How Models Behave 

If prompts are the instructions, then parameters are the one that control how the model follows them. Some important parameters to know include:

| Parameter                    | What It Controls                                  | Typical Use                                           |
| ---------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| **temperature**              | Randomness or creativity                          | Low for structured outputs, high for creative writing |
| **max_tokens / num_predict** | How long the output is                            | Prevent overly long responses                         |
| **top_p (nucleus sampling)** | How focused or broad the model's word choices are | Fine-tune randomness                                  |
| **top_k**                    | How many candidate words the model considers      | Mostly in open-source models                          |
| **repeat_penalty**           | Avoids repetition                                 | Stops the model from looping                          |
| **stop_sequences**           | Where generation stops                            | Useful for structured outputs                         |

These settings control how the model behaves in practice. They affect reliability, cost, and output quality.

## Why Prompt Behavior Matters

When using LLMs to build tools or apps, outputs can drift. A model that worked perfectly one day might produce overly creative responses or slightly broken JSON the next. Responses may slow down, and costs may increase unexpectedly.

Studying prompt behavior helps prevent these issues before they occur. By adjusting prompts, roles, temperature, and token limits, it becomes possible to spot potential problems early and produce more predictable results.

One approach that makes a big difference is **prompt roles**. Some platforms let you set roles directly (system, user, assistant), while others simulate roles by including them in the text. The goal is the same: providing context so the model behaves consistently.

**Example using roles:**

```python
messages = [
    {"role": "system", "content": "Always respond with valid JSON only using the format {\"result\": number}. No extra text."},
    {"role": "user", "content": "Calculate 12 x 8."}
]
```

**Example without roles:**

```python
messages = [
    {"role": "user", "content": "Calculate 12 × 8 and return the result as JSON with a key called result."}
]
```

**Tip:** Using roles often leads to more consistent behavior, especially for structured outputs, tone, and style.

## Temperature: Creativity vs Reliability

Temperature is one of the most important settings to understand. It controls how creative or predictable the model is.

- **At 0.0**, the model is precise, predictable, and safe. Ideal for structured outputs like JSON or repetitive API responses. Responses are reliable but can feel rigid.
- **At 0.5**, outputs start to feel more natural. Sentences flow better, with minor variations, balancing reliability and liveliness.
- **Above 1.0**, outputs become highly creative. This can produce clever phrasing, fresh ideas, and humor, but responses may also wander off-topic or have formatting issues.

**Key takeaway:** Temperature controls the balance between creativity and reliability. For production tasks, keep it low (0-0.3). For brainstorming or creative tasks, it can be increased, but do so intentionally.

## Extracting JSON Reliably

Producing valid JSON reliably can be tricky. Even with clear instructions, the model may add explanations, comments, or stray punctuation.

The best common approach it to combines this three strategies:

1. **System-level constraints**, for example: “Respond ONLY with valid JSON. No explanations.”
2. **Few-shot examples**, show the model exactly what a correct output should look like.
3. **Validation**, parse the output, catch errors, and handle them gracefully.

Using these together makes JSON output predictable and consistent, which is essential in production environments.

## Token Usage, Cost, Latency, and Context Limits

LLMs work with tokens, not words, and every token affects cost, response speed, and context limits.

Tracking token usage reveals how temperature, output length, and even small prompt changes affect costs and latency. Higher temperature often produces longer responses, increasing token usage and slightly slowing responses. A few hundred extra tokens per request may seem small, but across thousands of requests, it adds up quickly.

## Prompt Patterns That Make a Difference

After exploring how LLMs respond to different prompts, several patterns are known to improve model behavior consistently.

**1. Few-Shot Learning**

Providing a few examples of correct output can dramatically stabilize formatting. For example:

```python
messages = [
    {"role": "system", "content": "You convert text into structured JSON."},
    {"role": "user", "content": "Name: Alice, Age: 30"},
    {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
    {"role": "user", "content": "Name: Bob, Age: 25"}
]
```

Observation: Few-shot prompting is known to make formatting more reliable, even at higher temperatures. The model learns the expected output pattern from the examples, reducing errors and inconsistencies.

**2. Chain-of-Thought Reasoning**

Encouraging the model to think step by step improves reasoning for complex tasks:

```python
{"role": "system", "content": "Think step by step and explain your reasoning before answering."}
```

Observation: Step-by-step prompts generally improve accuracy and reasoning quality. The tradeoff is that token usage increases and output length becomes less predictable. This highlights a common balance when working with LLMs: accuracy versus cost versus output stability.

**3. Role-Based Prompts**

Assigning a role helps shape tone, depth, and perspective:

```python
{"role": "system", "content": "You are a senior backend engineer reviewing production code."}
```

Use cases: Code reviews, architecture analysis, and technical explanations benefit from role-based prompts. Giving the model a clear persona results in more consistent, targeted, and context-aware responses.

Using these patterns, including few-shot examples, chain-of-thought reasoning, and role-based instructions, can guide the model more predictably and improve output quality. Over time, these strategies help anticipate how changes to prompts affect results.

## Experiment Setup

The behavior described in this post is based on small, repeatable experiments rather than isolated prompt examples. A simple Python script was used to run the same prompts multiple times while varying temperature and prompt structure.

The full reference implementation is available here:  
<a href="https://github.com/reynaldineo/explore-rag/tree/main/llm-prompt-temperature-token" target="_blank" rel="noopener noreferrer">
  github.com/reynaldineo/explore-rag/tree/main/llm-prompt-temperature-token
</a>

The script runs against a local Ollama model and covers single-user prompts, role-based prompts, few-shot JSON extraction, chain-of-thought reasoning, and several edge cases. For each request, it records model output, token counts, and response latency in structured JSON logs.

This setup makes it easier to compare runs and observe how prompt design and temperature changes affect consistency and token usage.

## Conclusion

LLMs respond predictably when prompts and parameters are treated as control inputs rather than suggestions. Small changes in wording, temperature, or token limits can have large effects on output quality, cost, and latency.

This makes systematic testing essential. Few-shot examples, role-based instructions, and careful parameter tuning help stabilize behavior and reduce surprises in production systems.

LLMs are not black boxes. They respond to structure, constraints, and context in measurable ways. Understanding these patterns makes it possible to build systems that are both flexible and dependable.
