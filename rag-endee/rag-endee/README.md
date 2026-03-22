# RAG Knowledge Base — Powered by Endee Vector Database

> **Submission for:** ENDEE OC.41989.2026.58432  
> **Process:** Project Submission  
> **Technology Stack:** Python · Endee · Sentence-Transformers · Docker

---

## Project Overview & Problem Statement

Modern AI applications require access to up-to-date, domain-specific knowledge that pre-trained Large Language Models (LLMs) simply don't have baked in. LLMs also hallucinate — confidently producing incorrect facts.

**Retrieval Augmented Generation (RAG)** solves both problems by:
1. Storing your knowledge base as semantic vectors in a fast vector database
2. Retrieving the most relevant passages at query time
3. Grounding the LLM's answer in retrieved facts

This project implements a **complete, production-ready RAG pipeline** using **Endee** as the vector database backend.

---

## System Design & Technical Approach

```
User Query
    │
    ▼
┌─────────────────┐     encode (384-dim)    ┌───────────────────────┐
│  Query Encoder  │ ──────────────────────► │  Endee Vector DB      │
│ (MiniLM-L6-v2)  │                         │  (cosine, INT8, HNSW) │
└─────────────────┘ ◄─── top-K chunks ───── └───────────────────────┘
         │
         ▼
┌─────────────────┐
│  Prompt Builder │  (question + retrieved context)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  LLM  (GPT-4o / │  → Final grounded answer
│  Claude / Llama)│
└─────────────────┘
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | **Endee** | High-perf, open-source, single-node up to 1B vectors |
| Embedding | `all-MiniLM-L6-v2` | 384-dim, fast, high quality for semantic search |
| Precision | `INT8` | 4× memory saving with minimal accuracy loss |
| Distance | `cosine` | Best for normalised sentence embeddings |
| Chunking | Word-based, overlap=50 | Preserves sentence context across chunk boundaries |

---

## How Endee is Used

Endee acts as the **persistent semantic memory** of the system.

### 1 — Index creation
```python
from endee import Endee, Precision

client = Endee()                          # connect to local server
client.create_index(
    name="rag_knowledge_base",
    dimension=384,                        # matches MiniLM output
    space_type="cosine",
    precision=Precision.INT8,             # quantised storage
)
```

### 2 — Document ingestion (upsert)
```python
index = client.get_index("rag_knowledge_base")
index.upsert([
    {
        "id":     "doc1_chunk_0",
        "vector": embedding_vector,       # list[float], length=384
        "meta": {
            "title": "Transformer Architecture",
            "text":  "The Transformer is a deep learning model...",
            "doc_id": "doc1",
        }
    }
])
```

### 3 — Semantic retrieval
```python
query_vector = embedder.encode("How does attention work?").tolist()
results = index.query(vector=query_vector, top_k=5)
# returns [{id, similarity, meta}, ...]
```

---

## Project Structure

```
rag-endee/
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py      # Core RAG pipeline (ingestion + retrieval + prompt)
├── tests/
│   └── test_pipeline.py     # Unit tests (no live Endee required)
├── demo.py                  # End-to-end demo with sample knowledge base
├── search.py                # Interactive CLI semantic search tool
├── docker-compose.yml       # Start Endee with one command
├── requirements.txt
└── README.md
```

---

## Setup & Execution Instructions

### Prerequisites

- Python 3.10+
- Docker & Docker Compose v2

---

### Step 1 — Fork & Clone (mandatory per evaluation rules)

```bash
# 1. Star the repo: https://github.com/endee-io/endee
# 2. Fork it to your GitHub account
# 3. Clone YOUR fork
git clone https://github.com/<YOUR_USERNAME>/endee.git
cd endee
```

Then clone this project alongside it:

```bash
git clone https://github.com/<YOUR_USERNAME>/rag-endee.git
cd rag-endee
```

---

### Step 2 — Start the Endee Vector Database

```bash
docker compose up -d
```

Verify it's running:

```bash
curl http://localhost:8080/api/v1/index/list
# Expected: {"indexes": []}
```

Or open the dashboard at **http://localhost:8080**

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Run the demo

```bash
python demo.py
```

Expected output:
```
============================================================
  RAG Pipeline Demo – powered by Endee Vector Database
============================================================

[1] Ingesting knowledge base ...
    ✓ 5 chunks stored in Endee index 'rag_knowledge_base'

[2] Running semantic queries ...

Q: How does the Transformer architecture work?
   Latency : 0.048s
   Top context chunks:
     • [0.921] Transformer Architecture – The Transformer is a deep learning...
     • [0.783] Sentence Embeddings – Sentence embeddings are dense vector...
     • [0.712] Large Language Models – Large Language Models (LLMs) are...
...
```

---

### Step 5 — Interactive semantic search

```bash
python search.py
```

Or single-query mode:

```bash
python search.py "What is a vector database?"
```

---

### Step 6 — Run tests

```bash
pytest tests/ -v
```

---

### Optional — Connect a real LLM

Edit `src/rag_pipeline.py`, find the `# --- Optional: plug in your LLM here ---` block and uncomment the OpenAI section. Then:

```bash
export OPENAI_API_KEY=sk-...
python demo.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENDEE_HOST` | `http://localhost:8080/api/v1` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | `""` | Auth token (blank = open mode) |
| `OPENAI_API_KEY` | — | Optional: for LLM answer generation |

---

## Stopping Endee

```bash
docker compose down        # stop containers
docker compose down -v     # stop + remove stored data
```

---

## License

Apache 2.0 — see [Endee LICENSE](https://github.com/endee-io/endee/blob/master/LICENSE)
