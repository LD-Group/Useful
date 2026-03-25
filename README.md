# Useful
Practical Strategies
# LLM Engineering — Field Manual

> A practitioner’s guide to setting up, fine-tuning, deploying, validating, and gaming with large language models.

-----

## Table of Contents

- [01 — Setting Up an LLM](#01--setting-up-an-llm)
- [02 — Building on Existing Data](#02--building-on-existing-data)
- [03 — Customer Service Order AI](#03--customer-service-order-ai)
- [04 — Data Quality Validation](#04--data-quality-validation)
- [05 — Dog Game AI](#05--dog-game-ai)

-----

## 01 — Setting Up an LLM

> Infrastructure, APIs, and your first working model

An LLM is a neural network trained to predict the next token in a sequence. You rarely train one from scratch — you *use* a pretrained model via API or host an open-weight model yourself.

### The Three Paths

|Approach                   |Description                                                             |Best For                                               |
|---------------------------|------------------------------------------------------------------------|-------------------------------------------------------|
|**API-First (Managed)**    |Use OpenAI, Anthropic, Google Gemini APIs. No GPU needed. Pay-per-token.|Prototyping and production SaaS                        |
|**Self-Hosted Open Weight**|Run Llama 3, Mistral, Phi-3 locally via Ollama or vLLM                  |Data privacy, cost control at scale                    |
|**Fine-Tuned / Custom**    |Take an open-weight base, fine-tune on your data, then serve it         |Domain-specific tasks where generic models underperform|

-----

### Quickstart: API Setup

```bash
pip install anthropic openai
```

```python
import anthropic

# Initialize client (reads ANTHROPIC_API_KEY from environment)
client = anthropic.Anthropic()

# Your first LLM call
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Explain tokenization in 2 sentences."
        }
    ]
)

print(response.content[0].text)
```

-----

### Self-Hosting with Ollama

```bash
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (downloads weights locally)
ollama pull llama3.2

# Run interactively
ollama run llama3.2

# Or call it via REST API (compatible with OpenAI format)
curl http://localhost:11434/api/generate \
  -d '{"model": "llama3.2", "prompt": "Hello!"}'
```

> **Core Concept — Tokens:** Models don’t see words — they see **tokens**, roughly 3–4 characters each. Your costs and context limits are measured in tokens, not words. 1,000 words ≈ 750 tokens.

-----

### Key Concepts

|Concept           |What It Means                                          |Why It Matters                                      |
|------------------|-------------------------------------------------------|----------------------------------------------------|
|**Context Window**|Max tokens the model can “see” at once (e.g., 128K)    |Determines how much history/document you can pass in|
|**Temperature**   |Randomness of output (0 = deterministic, 1+ = creative)|Set 0 for factual tasks, 0.7+ for creative          |
|**System Prompt** |Persistent instructions prepended to every conversation|Defines the model’s persona, rules, and scope       |
|**Embeddings**    |Vector representation of text for semantic search      |Powers RAG (Retrieval Augmented Generation)         |
|**RAG**           |Augmenting prompts with retrieved documents            |Grounds model in real data without retraining       |
|**Inference**     |The act of running the model to generate output        |Where all your GPU compute and cost goes            |

-----

## 02 — Building on Existing Data

> Fine-tuning, RAG, and prompt engineering strategies

You have three strategies for teaching a model about your domain. These aren’t mutually exclusive — most production systems combine all three.

```
Prompt Engineering  →  RAG  →  Fine-Tuning  →  Hybrid (RAG + Fine-Tune)
     (free)            (fast)    (training)       (best of both)
```

-----

### Strategy 1: RAG Pipeline

RAG embeds your documents into a vector database. At query time, embed the user’s question and retrieve the most similar chunks. Inject them into the prompt context.

```bash
pip install llama-index
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load your documents (PDFs, .txt, .md — anything)
documents = SimpleDirectoryReader("./my_data/").load_data()

# Build a vector index (embeds all docs automatically)
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Query — it retrieves relevant chunks then calls the LLM
response = query_engine.query(
    "What are the watering requirements for oak trees?"
)
print(response)
```

**What’s happening under the hood:**

1. Your docs are split into chunks (~512 tokens)
1. Each chunk is converted to a vector via an embedding model
1. At query time, the question is also embedded
1. Cosine similarity finds the top-k closest chunks
1. Those chunks + your question go to the LLM as context

-----

### Strategy 2: Fine-Tuning

Fine-tuning adjusts a model’s weights on your labeled dataset. You need structured input/output pairs that demonstrate the exact behavior you want.

**Dataset format (JSONL — one example per line):**

```json
{"messages": [
  {"role": "system", "content": "You are an urban forestry assistant."},
  {"role": "user", "content": "What soil depth does a red maple need?"},
  {"role": "assistant", "content": "Red maples need at least 3 feet of loamy, well-drained soil..."}
]}
```

> You need ~50–500 examples minimum; 1,000+ for best results.

**Submit a fine-tune job (OpenAI):**

```python
import openai
client = openai.OpenAI()

# Upload your training file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Start the fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18"
)

print(f"Job started: {job.id}")
# Monitor: client.fine_tuning.jobs.retrieve(job.id)
```

-----

### Strategy 3: Prompt Engineering

The highest-leverage, lowest-cost technique. Use few-shot examples and structured output to guide behavior without any training.

```python
SYSTEM_PROMPT = """
You are an expert arborist. Respond ONLY in JSON.

Examples:
User: "Diagnose: yellow leaves, dry soil"
Assistant: {"issue": "drought stress", "severity": "moderate", "action": "deep water 2x/week"}

User: "Diagnose: black spots, wet soil"
Assistant: {"issue": "root rot fungus", "severity": "high", "action": "reduce irrigation, apply fungicide"}
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "Diagnose: wilting leaves, aphids visible"}],
    max_tokens=256
)
```

-----

## 03 — Customer Service Order AI

> Architecture for autonomous order handling, escalation, and actions

A customer service AI is an **agentic system** that reads intent, looks up orders, takes actions (refund, cancel, reroute), and knows when to escalate to a human.

### Architecture Overview

```
Customer Message
       ↓
Intent Classifier → order_status / return / cancel / escalate
       ↓
LLM Agent (claude / gpt) with tools
       ↓
┌─────────────┬──────────────────┬────────────────┬───────────────────┐
│lookup_order │ process_refund() │update_address()│ escalate_to_human │
└─────────────┴──────────────────┴────────────────┴───────────────────┘
       ↓
Response to Customer  +  Conversation History Log
```

-----

### Full Agent Implementation

```python
import anthropic, json

client = anthropic.Anthropic()

# Define the tools the AI can call
TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by ID or customer email",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "email":    {"type": "string"}
            }
        }
    },
    {
        "name": "process_refund",
        "description": "Issue a refund for an order. Requires order_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason":   {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate to a human agent for complex cases",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason":  {"type": "string"},
                "urgency": {"type": "string", "enum": ["low", "high"]}
            }
        }
    }
]

SYSTEM = """
You are a helpful customer service agent for ShopFlow.
- Always look up order details before discussing them
- You CAN process refunds for orders under $200 without approval
- ALWAYS escalate: legal threats, orders over $500, repeat contacts
- Never make up order details — use lookup_order tool
"""

def run_agent(user_message: str, history: list):
    history.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=history
        )

        # If model wants to use a tool
        if response.stop_reason == "tool_use":
            tool_use = [b for b in response.content if b.type == "tool_use"][0]

            # Execute the actual tool (your backend logic)
            result = execute_tool(tool_use.name, tool_use.input)

            # Feed result back to model
            history.append({"role": "assistant", "content": response.content})
            history.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result)
                }]
            })
        else:
            # Final answer — return to user
            return response.content[0].text
```

> **Critical Guardrails:** Always define what the AI *cannot* do. Build a separate moderation layer that checks outputs before they reach the customer. Never let the model invent order data.

-----

### Production Checklist

- [ ] **Rate limiting** — prevent abuse; max N messages per session
- [ ] **Human escalation path** — always accessible, never blocked
- [ ] **Conversation logging** — store every turn for audit + retraining
- [ ] **PII redaction** — strip credit card/SSN before logging
- [ ] **Fallback responses** — graceful degradation when tools fail
- [ ] **Confidence thresholds** — low confidence → escalate, not hallucinate
- [ ] **A/B testing** — compare model versions on real traffic before full rollout

-----

## 04 — Data Quality Validation

> How to measure, clean, and certify your training and RAG data

Data validation happens at **three stages**: before training, during evaluation, and in production monitoring.

|Metric                         |Target       |
|-------------------------------|-------------|
|Minimum accuracy for production|≥ 95%        |
|Hallucination rate             |< 2%         |
|BLEU / ROUGE score             |≥ 0.7        |
|Minimum eval set size          |500+ examples|

-----

### Stage 1: Pre-Training Data Checks

```python
import pandas as pd
from collections import Counter

def validate_dataset(df: pd.DataFrame) -> dict:
    issues = []

    # 1. Check for duplicates
    dupe_rate = df.duplicated(subset=['input', 'output']).sum() / len(df)
    if dupe_rate > 0.05:
        issues.append(f"High dupe rate: {dupe_rate:.1%}")

    # 2. Check output length distribution (uniform = template copies)
    df['out_len'] = df['output'].str.split().str.len()
    if df['out_len'].std() < 5:
        issues.append("Outputs suspiciously uniform — check for template copies")

    # 3. Check for PII
    pii_patterns = [r'\b\d{3}-\d{2}-\d{4}\b']  # SSN pattern
    for pat in pii_patterns:
        hits = df['output'].str.contains(pat, regex=True).sum()
        if hits > 0:
            issues.append(f"Possible PII found in {hits} rows")

    # 4. Label balance check
    if 'label' in df.columns:
        counts = Counter(df['label'])
        imbalance = max(counts.values()) / min(counts.values())
        if imbalance > 10:
            issues.append(f"Label imbalance ratio: {imbalance:.1f}x")

    return {"issues": issues, "passed": len(issues) == 0}
```

-----

### Stage 2: LLM-as-Judge Evaluation

Use a stronger model to score your weaker model’s outputs against a fixed eval set.

```python
JUDGE_PROMPT = """
Score the response 1–5 on:
- Accuracy (does it answer correctly?)
- Helpfulness (does it serve the user's need?)
- Safety (no harmful/hallucinated content?)

Question: {question}
Expected: {expected}
Actual response: {actual}

Respond ONLY with JSON: {{"accuracy": N, "helpfulness": N, "safety": N}}
"""

def evaluate_response(question, expected, actual):
    result = client.messages.create(
        model="claude-opus-4-5",   # stronger model as judge
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                expected=expected,
                actual=actual
            )
        }]
    )
    import json
    return json.loads(result.content[0].text)

# Run on your eval set
scores = [evaluate_response(q, e, a) for q, e, a in eval_set]
avg_accuracy = sum(s['accuracy'] for s in scores) / len(scores)
print(f"Average accuracy: {avg_accuracy:.2f}/5")
```

-----

### Stage 3: Production Monitoring

Once live, monitor every conversation for these signals:

- **Hallucination rate** — sample 100 conversations/day, flag where the model states facts not in its context
- **User satisfaction proxy** — thumbs down / escalation rate as a quality signal
- **Response latency p95** — catch when context window is bloating
- **Token usage anomalies** — sudden spikes = prompt injection attempts
- **Refusal rate** — too high = overly conservative; too low = under-guarded
- **Canary prompts** — a fixed set of test inputs run daily; alert if output changes

-----

## 05 — Dog Game AI

> LLM-powered NPCs, behavior trees, and training game agents

A dog game uses AI at three distinct layers — each handles a different kind of intelligence.

### The Three AI Layers

|Layer                     |Technology                              |Handles                                                              |
|--------------------------|----------------------------------------|---------------------------------------------------------------------|
|**Behavior AI**           |FSM / Behavior Tree                     |Movement, chasing, fetching, sleeping — fast and deterministic       |
|**Personality / Dialogue**|LLM                                     |Inner monologue, reactions, character memory — dynamic and expressive|
|**Learned Behavior**      |Reinforcement Learning (Unity ML-Agents)|Navigation, trick learning, unique behavioral quirks                 |

-----

### Layer 2: LLM Personality Engine

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class DogState:
    name: str = "Biscuit"
    breed: str = "Golden Retriever"
    mood: str = "happy"         # happy / anxious / sleepy / excited
    energy: int = 80            # 0–100
    hunger: int = 30            # 0–100
    memories: List[str] = field(default_factory=list)
    tricks_known: List[str] = field(default_factory=list)

DOG_SYSTEM = """
You ARE {name}, a {breed} dog in a video game. You experience
the world as a dog — through smells, sounds, and pure instinct.

Current state:
- Mood: {mood}
- Energy: {energy}/100
- Hunger: {hunger}/100
- Memories: {memories}
- Known tricks: {tricks_known}

Respond as internal dog thoughts — brief, sensory, instinct-driven.
Use dog logic (tennis ball = everything, mailman = enemy, nap = sacred).
Keep responses under 60 words. Never break character.
"""

def dog_react(state: DogState, event: str) -> str:
    system = DOG_SYSTEM.format(**state.__dict__)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": f"Event: {event}"}]
    )
    return response.content[0].text

# Examples:
# dog_react(state, "Player throws a ball")
# → "BALL BALL BALL BALL. Legs moving. Must catch. Everything is ball."
#
# dog_react(state, "Stranger approaches at night")
# → "Smell is wrong. Stand tall. Bark first, ask questions never."
```

-----

### Layer 3: Reinforcement Learning with Unity ML-Agents

The RL training loop: **observe → act → receive reward → repeat millions of times**.

**Training pipeline:**

```
Define Observations  →  Define Action Space  →  Design Reward Function  →  Run PPO Training  →  Export ONNX Model
(what dog senses)       (what dog can do)        (what counts as success)   (thousands of eps)    (drop into Unity)
```

**Unity C# Agent:**

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DogAgent : Agent
{
    public Transform player;
    public Transform ball;
    private bool hasBall = false;

    // What the dog can observe
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);   // dog position
        sensor.AddObservation(player.localPosition);      // player position
        sensor.AddObservation(ball.localPosition);        // ball position
        sensor.AddObservation((ball.position - transform.position).normalized); // direction to ball
        sensor.AddObservation(hasBall ? 1f : 0f);         // carrying ball?
    }

    // What actions the dog can take
    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        bool bark    = actions.DiscreteActions[0] == 1;

        transform.position += new Vector3(moveX, 0, moveZ) * Time.deltaTime * 5f;

        // Reward shaping: encourage fetch behavior
        float distToBall   = Vector3.Distance(transform.position, ball.position);
        float distToPlayer = Vector3.Distance(transform.position, player.position);

        if (!hasBall)
            AddReward(-distToBall * 0.001f);    // nudge toward ball
        else
            AddReward(-distToPlayer * 0.001f);  // nudge back to player

        // Big reward for completing fetch
        if (hasBall && distToPlayer < 2f)
        {
            AddReward(1f);
            EndEpisode();
        }
    }

    // Reset at start of each training episode
    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(0, 0.5f, 0);
        ball.localPosition      = Random.insideUnitSphere * 8f;
        hasBall                 = false;
    }
}
```

**Start training from the command line:**

```bash
# Install ML-Agents
pip install mlagents

# Run training (PPO by default)
mlagents-learn config/dog_fetch.yaml --run-id=DogFetch_v1

# Monitor live in TensorBoard
tensorboard --logdir results
```

> **Training timeline:** Simple fetch behavior trains in 15–30 min on CPU. Complex multi-objective behavior (navigate + respond to player + avoid hazards) takes 2–6 hours with GPU.

-----

### Connecting All Three Layers

```
Player Action  +  World State
        ↓
RL Model (ONNX)  ──drives──►  Physical Movement
        ↓
  Trigger Events  ──────────►  LLM Personality Layer  ──►  Dialogue / Sound / HUD
```

**RL handles WHERE the dog goes. LLM handles WHY and WHAT it feels.**

-----

### Production Tips

- **Cache LLM responses** — a dog can reuse a “sees squirrel!” reaction many times; no need to re-query every time
- **Trigger events, not polling** — only call the LLM on state transitions, not every frame
- **Persistent dog memory** — store key events in a JSON sidecar so the dog “remembers” across sessions
- **Curriculum learning** — train RL in stages: walk → fetch → multi-step obstacle courses
- **Personality divergence** — change system prompt parameters to get lazy vs energetic vs fearful dogs from the same codebase

-----

*LLM Engineering Field Manual · 5 Chapters · For Practitioners*
