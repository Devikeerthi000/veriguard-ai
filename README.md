<h1 align="center">VeriGuard AI</h1>

<p align="center">
  <strong>Enterprise-Grade LLM Hallucination Detection & Verification Engine</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#how-it-works">How It Works</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Groq-LLM-purple.svg" alt="Groq LLM"/>
</p>

---

## Overview

VeriGuard AI is a production-ready hallucination detection system that verifies factual claims in AI-generated content against a comprehensive knowledge base.

### The Problem

Large Language Models generate confident-sounding but factually incorrect content:
- Fabricated statistics and numerical values
- Invented historical events and dates
- False scientific claims
- Incorrect geographic and legal information

### The Solution

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   LLM Output    │────▶│   VeriGuard AI   │────▶│ Verified Output │
│  (unverified)   │     │   Verification   │     │ + Risk Report   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Stage Verification** | Claim extraction → Evidence retrieval → LLM verification → Risk scoring |
| **Semantic Search** | FAISS-powered vector search with cross-encoder reranking |
| **Contradiction Detection** | Identifies conflicts between evidence sources |
| **Temporal Analysis** | Validates date/time claims against historical records |
| **Numerical Precision** | Detects statistical and numerical discrepancies |
| **Risk Quantification** | Severity scoring with actionable recommendations |

### Technical Highlights

- **FastAPI REST API** with OpenAPI documentation
- **500+ fact knowledge base** across science, medicine, finance, history
- **Multi-backend caching** (memory, disk, Redis)
- **Production logging** with JSON structured output
- **Batch processing** for high-throughput scenarios
- **Configurable verification depth** (quick/standard/thorough)

---

## Usage

### Interactive CLI

```bash
python main.py
```

```
============================================================
  VeriGuard AI - Hallucination Detection Engine v2.0.0
============================================================

Enter text to verify (type 'quit' to exit, 'help' for commands):

>>> The capital of Australia is Sydney

------------------------------------------------------------
VERIFICATION RESULTS
------------------------------------------------------------
Request ID: a1b2c3d4
Total Claims: 1
Hallucination Rate: 100.0%
Risk Score: 0.85

CLAIMS:

  1. [FALSE] The capital of Australia is Sydney
     Status: CONTRADICTED
     Confidence: 95%
     Risk: HIGH (0.85)
     Reason: The capital of Australia is Canberra, not Sydney.
============================================================
```

### REST API

```bash
# Start API server
python main.py --api

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Single Verification

```bash
python main.py --verify "Einstein won the Nobel Prize for relativity"
```

### Rebuild Knowledge Base

```bash
python main.py --rebuild
```

---

## API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/verify` | Verify text for hallucinations |
| `POST` | `/verify/quick` | Quick verification |
| `POST` | `/verify/batch` | Batch verification |
| `GET` | `/knowledge/stats` | Knowledge base statistics |
| `POST` | `/knowledge/search` | Semantic search |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Einstein won the Nobel Prize for relativity in 1921.",
    "extraction_mode": "standard",
    "verification_depth": "standard"
  }'
```

### Example Response

```json
{
  "request_id": "a1b2c3d4",
  "total_claims": 1,
  "hallucination_rate": 1.0,
  "overall_risk_score": 0.85,
  "analyses": [
    {
      "claim": {
        "text": "Einstein won the Nobel Prize for relativity"
      },
      "verification": {
        "status": "CONTRADICTED",
        "confidence": 0.95,
        "explanation": "Einstein won the Nobel Prize for the photoelectric effect, not relativity."
      },
      "risk": {
        "severity": "HIGH",
        "risk_score": 0.85
      }
    }
  ]
}
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              VeriGuard AI Pipeline                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Input     │───▶│   Claim     │───▶│  Evidence   │───▶│   Claim     │   │
│  │   Text      │    │  Extractor  │    │  Retriever  │    │  Verifier   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                            │                  │                  │          │
│                            ▼                  ▼                  ▼          │
│                     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│                     │  Groq LLM   │    │ FAISS Index │    │  Groq LLM   │   │
│                     │ (llama-3.3) │    │ + Reranker  │    │ (llama-3.3) │   │
│                     └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Risk Assessment Engine                          ││
│  │  • Status-based scoring  • Domain sensitivity  • Numerical precision   ││
│  │  • Temporal consistency  • Contradiction detection  • Recommendations  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│                            ┌─────────────────┐                              │
│                            │ Verification    │                              │
│                            │ Report + Risk   │                              │
│                            │ Assessment      │                              │
│                            └─────────────────┘                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
veriguard-ai/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
│
├── api/                   # REST API layer
│   ├── app.py            # FastAPI application
│   └── routes/           # API endpoints
│
├── core/                  # Core verification engine
│   ├── models.py         # Pydantic data models
│   ├── pipeline.py       # Main orchestration
│   ├── extractor.py      # Claim extraction (Groq LLM)
│   ├── embedder.py       # Embedding generation (Sentence Transformers)
│   ├── index.py          # FAISS vector indexing
│   ├── retriever.py      # Evidence retrieval + reranking
│   ├── verifier.py       # Claim verification (Groq LLM)
│   └── risk_engine.py    # Risk assessment
│
├── config/               # Configuration
│   └── settings.py       # Pydantic settings
│
├── utils/                # Utilities
│   ├── logger.py         # Structured logging
│   └── cache.py          # Caching system
│
└── data/                 # Knowledge base
    └── knowledge_base/   # 500+ categorized facts
```

---

## How It Works

### Step-by-Step Process

1. **Claim Extraction** - Groq LLM extracts factual claims from input text
2. **Embedding** - Sentence Transformer converts claims to 768-dimensional vectors
3. **Evidence Retrieval** - FAISS finds semantically similar facts from knowledge base
4. **Reranking** - Cross-encoder reranks results for precision
5. **Verification** - Groq LLM compares claim against evidence
6. **Risk Scoring** - Multi-factor risk assessment with severity levels

### Verification Statuses

| Status | Meaning |
|--------|---------|
| `SUPPORTED` | Evidence confirms the claim |
| `CONTRADICTED` | Evidence conflicts with the claim |
| `PARTIALLY_SUPPORTED` | Some aspects true, others false |
| `UNVERIFIABLE` | Relevant evidence but insufficient |
| `INSUFFICIENT_EVIDENCE` | No relevant evidence found |

### Risk Severity Levels

| Severity | Description |
|----------|-------------|
| `CRITICAL` | Do not publish - requires expert review |
| `HIGH` | Significant concerns - manual verification needed |
| `MEDIUM` | Some uncertainty - consider adding qualifiers |
| `LOW` | Minor concerns - generally reliable |
| `NEGLIGIBLE` | Verified accurate |

---

## Knowledge Base

| Category | Topics |
|----------|--------|
| Science & Technology | Physics, Chemistry, AI, Quantum Computing, Climate |
| Medicine & Health | Clinical trials, Pharmacology, Neuroscience, Genetics |
| Business & Finance | Economics, Banking, Cryptocurrency, Taxation |
| History & Culture | World History, Geography, Art, Philosophy |

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (required) |
| `LOG_LEVEL` | Logging level (default: INFO) |
| `API_PORT` | API server port (default: 8000) |
| `CACHE_BACKEND` | Cache type: memory/disk/redis |

### Verification Modes

| Mode | Use Case |
|------|----------|
| `strict` | Legal, medical content - only explicit facts |
| `standard` | General use - balanced extraction |
| `permissive` | Research - includes implicit claims |

---

## Technologies Used

| Component | Technology |
|-----------|------------|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | Sentence Transformers (all-mpnet-base-v2) |
| Vector Search | FAISS |
| Reranking | Cross-Encoder |
| API | FastAPI |
| Configuration | Pydantic Settings |

---

## Author

**Devi Keerthi Adapa**

- GitHub: [@Devikeerthi000](https://github.com/Devikeerthi000)
- Email: keerthiadapa70@gmail.com
