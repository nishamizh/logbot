# LogBot 🤖

> Production-grade LLM-powered log analysis and anomaly detection system.

Built as a demonstration of senior engineering judgment across the full stack:
multi-layer AI architecture, real-time anomaly detection, ReAct agent loop,
production guardrails, and a FastAPI serving layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        API Layer                         │
│  FastAPI  ·  Request-ID middleware  ·  Pydantic models  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     Agent Layer                          │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Planner   │  │ StateMachine │  │   Guardrails  │  │
│  │  ReAct loop │  │  FSM 8-state │  │ PII · Inject  │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │              Tool Registry (5 tools)             │   │
│  │  search_logs · get_anomalies · analyze_window   │   │
│  │  summarize_logs · get_service_health            │   │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼──────────┐   ┌─────────────▼──────────────┐
│   Detection Layer  │   │      Retrieval Layer        │
│                    │   │                             │
│  IsolationForest   │   │  PySpark pipeline           │
│  ZScore (Welford)  │   │  HuggingFace embeddings     │
│  ModelRegistry     │   │  ChromaDB vector store      │
│  AlertBuffer       │   │  Log preprocessor           │
└────────────────────┘   └─────────────────────────────┘
          │
┌─────────▼──────────┐
│     Core Layer     │
│  Pydantic config   │
│  structlog JSON    │
│  TimedBlock        │
└────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| ReAct agent loop | THOUGHT→ACTION→OBSERVATION→ANSWER gives interpretable, auditable reasoning |
| Dual anomaly detectors | IsolationForest catches structural outliers; ZScore (Welford) catches rate spikes |
| Guardrail chains | Input PII scrub + injection detection; output format + hallucination checks |
| Explicit FSM | 8-state machine makes every transition visible and testable |
| `lru_cache` singletons | One config parse per process; safe to import anywhere |
| `ContextVar` request IDs | Async-safe trace propagation without threading.local |
| Score-first Welford | Update running stats after scoring to prevent self-dilution |
| Stub mode everywhere | Every layer works without real dependencies — demo without GPU/ChromaDB |

---

## Project structure

```
logbot/
├── logbot/
│   ├── agent/
│   │   ├── planner.py        ← ReAct loop orchestrator
│   │   ├── tools.py          ← Tool definitions + ToolRegistry
│   │   ├── guardrails.py     ← Input/output validation, PII scrubbing
│   │   ├── state_machine.py  ← 8-state FSM
│   │   └── prompts.py        ← All LLM prompt templates
│   ├── retrieval/
│   │   ├── spark_pipeline.py ← PySpark batch ingestion
│   │   ├── preprocess.py     ← Raw log → structured fields
│   │   ├── embedder.py       ← HuggingFace sentence-transformers
│   │   └── vector_store.py   ← ChromaDB wrapper
│   ├── detection/
│   │   ├── anomaly_detector.py ← Orchestration + AlertBuffer
│   │   └── model_loader.py     ← IsolationForest + ZScore lifecycle
│   ├── api/
│   │   └── server.py         ← FastAPI app
│   ├── core/
│   │   ├── config.py         ← Pydantic Settings (all env vars)
│   │   └── logging.py        ← structlog JSON + TimedBlock
│   └── main.py               ← Entrypoint
├── tests/
├── data/
├── scripts/
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/nishamizh/logbot
cd logbot
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — minimum required for stub mode: nothing!
# For real LLM: set LLM_HF_TOKEN or OPENAI_API_KEY
```

### 3. Run

```bash
# Start the API server
python -m logbot.main

# Or just check config
python -m logbot.main --check
```

Server starts at `http://localhost:8080`

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + readiness probe |
| `GET` | `/metrics` | Request counters, anomaly counts |
| `POST` | `/analyze` | Ask a natural language question about logs |
| `POST` | `/anomalies/detect` | Run anomaly detection on a log window |
| `GET` | `/anomalies/recent` | Recent alerts from the alert buffer |
| `POST` | `/logs/ingest` | Ingest log entries + auto-detect anomalies |
| `GET` | `/agent/tools` | List agent tools and parameter schemas |
| `GET` | `/docs` | Swagger UI |

### Example: Ask a question

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is the payments service showing high error rates?"}'
```

```json
{
  "session_id": "a1b2c3d4",
  "answer": "**Severity**: CRITICAL\n**Root Cause**: DB connection pool exhaustion\n**Recommended Actions**:\n1. Increase pool size\n2. Add retry with backoff",
  "success": true,
  "iterations": 2,
  "tool_calls": [
    {"tool": "get_anomalies", "success": true, "elapsed_ms": 1.2},
    {"tool": "search_logs",   "success": true, "elapsed_ms": 0.8}
  ],
  "elapsed_ms": 1847.3
}
```

### Example: Ingest logs

```bash
curl -X POST http://localhost:8080/logs/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "payments.log",
    "entries": [
      {"level": "ERROR",    "service": "payments", "message": "DB timeout after 30s"},
      {"level": "CRITICAL", "service": "payments", "message": "Circuit breaker OPEN"},
      {"level": "INFO",     "service": "payments", "message": "Health check OK"}
    ]
  }'
```

---

## Anomaly detection

LogBot uses two complementary detectors:

**IsolationForest** (structural outliers)
- Fits on historical log windows (error rates, service counts, burst patterns)
- Detects multi-dimensional anomalies — e.g. high error rate + low volume + many unique services simultaneously
- Requires training data; loaded from `data/models/` on startup

**ZScore with Welford's online algorithm**
- Univariate, per-metric, O(1) memory — no history buffer
- Scores each metric *before* updating running stats (prevents self-dilution)
- Works immediately with no training data; effective after ~30 windows

Both detectors feed into a fusion layer that emits a single `Alert` with severity, contributing features, and a human-readable summary.

---

## Agent loop (ReAct)

```
User question
     │
     ▼
Input Guardrails ──── blocked? ──→ GuardrailResponse
     │
     ▼
PLANNING (LLM generates THOUGHT + ACTION)
     │
     ▼
EXECUTING (ToolRegistry.execute)
     │
     ▼
REFLECTING (inject OBSERVATION, decide: loop or respond)
     │
     ▼
RESPONDING (LLM generates final ANSWER)
     │
     ▼
Final Guardrails (PII scrub, harmful content check)
     │
     ▼
AgentResponse → API
```

Max 5 iterations by default. On exhaustion, forces a RESPONDING step with accumulated context.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Environment variables

See `.env.example` for the full list. Key variables:

```bash
ENVIRONMENT=development          # development | staging | production
LOG_LEVEL=INFO
LLM_PROVIDER=huggingface         # huggingface | openai | anthropic
LLM_HF_TOKEN=hf_...             # required for gated models
OPENAI_API_KEY=sk-...           # if using OpenAI
CHROMA_HOST=localhost            # ChromaDB host
API_PORT=8080
```

---

## Built with

- **FastAPI** — async API framework
- **Pydantic v2** — config, request/response validation
- **structlog** — structured JSON logging
- **HuggingFace Transformers** — LLM inference
- **sentence-transformers** — log embeddings
- **ChromaDB** — vector store
- **scikit-learn** — IsolationForest anomaly detection
- **PySpark** — log ingestion pipeline
- **uvicorn** — ASGI server

---

*LogBot — built by Nisha Mizhquiri*
