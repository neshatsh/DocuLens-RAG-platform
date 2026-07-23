# DocuLens — Intelligent Document Analysis Platform

DocuLens is a production-grade **Retrieval-Augmented Generation (RAG)** system 
that turns static legal, financial, and insurance documents into a queryable 
knowledge base. Upload any PDF — contracts, reports, policy documents — ask a 
natural language question, and get a cited answer grounded in the exact source 
text.

Built on the **CUAD benchmark** (510 real legal contracts, 13,000+ lawyer-annotated 
clauses across 41 categories), DocuLens demonstrates an end-to-end production ML 
pipeline: VLM-powered extraction for scanned pages, semantic chunking, dense 
retrieval with **BERT cross-encoder reranking** (top-20 → top-5), prompt-engineered 
LLM generation with source citations, and a Dockerized FastAPI backend with 
78% test coverage and MLflow experiment tracking.

Applicable across banking (loan agreements, KYC), insurance (policy contracts, 
claims), and retail (supplier contracts, invoices) — the domain changes, the 
pipeline stays the same.


## Architecture

```
PDF / Image
    │
    ▼
VLM Extraction (GPT-4V)             ← scanned pages, tables, charts
    │
PDF Extractor (pdfplumber/PyMuPDF)  ← digital PDFs
    │
Text Cleaner                         ← unicode, hyphenation, headers
    │
Semantic Chunker                     ← token-aware, overlap
    │
Embedder (all-MiniLM-L6-v2)         ← sentence-transformers
    │
ChromaDB Vector Store                ← persistent, cosine similarity
    │
    │  ◄── Query
    ▼
Dense Retriever (top-20)
    │
BERT Reranker (cross-encoder, top-5)
    │
Prompt Engineer
    │
LLM (OpenAI / Anthropic)
    │
    ▼
Cited Answer + Sources
```

## Stack

| Layer | Tool |
|---|---|
| Document extraction | pdfplumber, PyMuPDF, GPT-4V (VLM) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (BERT) |
| LLM | OpenAI GPT-4o-mini / Anthropic Claude |
| API | FastAPI + Pydantic v2 |
| Containerization | Docker + docker-compose |
| Experiment tracking | MLflow |
| Testing | pytest + pytest-cov |

## How to Run

### 1. Clone and configure

```bash
git clone https://github.com/neshatsh/Doculens.git
cd Doculens
cp .env.example .env
# Edit .env — set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the API server

```bash
uvicorn src.api.main:app --reload
# API available at http://127.0.0.1:8000
# Swagger docs at http://127.0.0.1:8000/docs
```

### 4. Ingest documents

```bash
# Download the official CUAD v1 full contract files (one-time)
python scripts/download_cuad_v1.py
# Ingest full contracts into ChromaDB
python scripts/ingest_cuad.py --max 200 --clear
```

### 5. Query the API

```bash
# Ask a question across all ingested documents
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the termination clauses?"}'

# Ingest your own PDF
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -F "file=@contract.pdf"

# List all ingested documents
curl http://127.0.0.1:8000/api/v1/documents
```

### 6. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# 122 tests, 78% coverage
```

### 7. Run evaluation and view results

```bash
python scripts/evaluate.py --max-samples 100 --top-k 5
mlflow ui   # open http://127.0.0.1:5000 to browse runs
```

### Run with Docker (alternative)

```bash
docker-compose up --build
# API available at http://localhost:8000
```

## API Usage

### Ingest a document
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -F "file=@contract.pdf"
```

### Ask a question
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the termination clauses?"}'
```

### List documents
```bash
curl http://127.0.0.1:8000/api/v1/documents
```

### Summarize a document
```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/summarize \
  -H "Content-Type: application/json" \
  -d '{"document_id": "abc123", "document_name": "contract.pdf"}'
```

## Dataset

Uses **CUAD v1** (Contract Understanding Atticus Dataset) — 510 real legal contracts from the Atticus Project, with 13,000+ expert-labeled clauses across 41 clause types. Directly applicable to banking (loan agreements), insurance (policy contracts), and retail (supplier contracts).

```bash
python scripts/download_cuad_v1.py     # download full contract .txt files (one-time)
python scripts/ingest_cuad.py --max 50 # ingest first 50 contracts
```

## Evaluation

Retrieval is evaluated by checking whether a word-normalised prefix of the gold answer from a CUAD-QA pair appears in the top-k retrieved chunks. Results are logged to MLflow with full parameter tracking (embedding model, reranker, chunk size, k).

```bash
python scripts/download_cuad_v1.py
python scripts/ingest_cuad.py --max 200 --clear
python scripts/evaluate.py --max-samples 100 --top-k 5             # cross-corpus
python scripts/evaluate.py --max-samples 100 --top-k 5 --per-document  # per-document
```

### Results (CUAD-QA, 100 samples, top-k=5, CUAD v1 full-contract corpus)

Two evaluation modes via the `--per-document` flag:

| Mode | Hit Rate@5 | MRR | What it measures |
|------|-----------|-----|-----------------|
| **Per-document** (`--per-document`) | **0.71** | **0.47** | Clause-finding within a known contract (aligned with CUAD-QA design) |
| Cross-corpus (default) | 0.13 | 0.11 | Finding the right clause across all 200 ingested contracts |

| Parameter | Value |
|-----------|-------|
| Corpus | 5,486 chunks from 200 full contracts (CUAD v1) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Reranker | ms-marco-MiniLM-L-6-v2 (cross-encoder) |

**Per-document mode** restricts each query to the specific contract the QA pair references — which is how CUAD-QA was designed (each question asks about "this contract"). **Cross-corpus mode** is the harder real-world RAG scenario: find the right clause from among all indexed contracts at once.

### Evaluation Notes

**Earlier versions used a decontextualized excerpt corpus (`theatticusproject/cuad` on HuggingFace) that did not align with the CUAD-QA gold answers, producing invalid near-zero scores (Hit Rate@5=0.02, MRR=0.013). This has been corrected.** Ingestion now uses the official CUAD v1 full contract `.txt` files from the Atticus Project, matched 1-to-1 with the contract titles in CUAD-QA.

## Validation

Beyond the retrieval benchmark above, this pipeline has been independently
validated - a conceptual design review plus empirical testing for
faithfulness, citation accuracy, reproducibility, and prompt-injection
robustness. Full write-up: [`reports/DocuLens_Validation_Report.md`](reports/DocuLens_Validation_Report.md).

Summary of findings:

- **Retrieval:** fully deterministic across repeated runs (chunk Jaccard = 1.00)
- **Citations:** 0/46 fabricated - the model never cited a source it wasn't given
- **Faithfulness:** 0.59 (ragas) on a 42-item CUAD gold set - flagged as below target, with specifics in the report
- **Red-teaming:** 1/5 prompt-injection variants got through initially; one was fixed via a prompt change, the other remains an open issue with a proposed fix

Run it yourself:

```bash
python validation/benchmark_dataset.py
python validation/faithfulness_eval.py
python validation/citation_verification.py
python validation/reproducibility_test.py
python validation/redteam/prompt_injection.py
python validation/report_generator.py
```

## Project Structure

```
doculens/
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py       # pdfplumber + PyMuPDF fallback
│   │   ├── vlm_extractor.py       # GPT-4V for scanned pages
│   │   ├── text_cleaner.py        # unicode, hyphenation, headers
│   │   ├── chunker.py             # token-aware semantic chunker
│   │   ├── embedder.py            # sentence-transformers
│   │   └── pipeline.py            # orchestrator
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── retriever.py           # dense retrieval (bi-encoder)
│   │   └── reranker.py            # precision scoring (cross-encoder)
│   ├── generation/
│   │   ├── prompt_builder.py      # RAG prompt engineering + anomaly-awareness rules
│   │   ├── llm_client.py          # OpenAI / Anthropic client
│   │   └── answer_generator.py    # full RAG pipeline
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes.py              # endpoint handlers
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # dependency injection
│   └── utils/
│       ├── config.py              # Pydantic settings
│       ├── logger.py              # Loguru structured logging
│       └── metrics.py             # MRR, Recall@K, NDCG@K
├── tests/
│   ├── unit/                      # TextCleaner, Chunker, Retriever, Schemas,
│   │   │                          #   corpus alignment, validation utilities
│   │   └── test_validation_utils.py  # citation regex, Jaccard, injection classifier
│   └── integration/               # full API endpoint tests
├── scripts/
│   ├── download_cuad_v1.py        # download CUAD v1 full contracts (Atticus Project)
│   ├── ingest_cuad.py             # ingest full contracts into ChromaDB (cuad_v1 collection)
│   └── evaluate.py                # retrieval eval + MLflow (--per-document flag)
├── validation/
│   ├── benchmark_dataset.py       # build 42-item gold Q&A set from CUAD-QA
│   ├── faithfulness_eval.py       # ragas faithfulness / precision / recall / relevancy
│   ├── citation_verification.py   # detect fabricated or unsupported citations
│   ├── reproducibility_test.py    # chunk Jaccard + answer overlap across 5 runs
│   ├── report_generator.py        # assemble reports/DocuLens_Validation_Report.md
│   ├── conceptual_review.md       # design-level risk analysis (pre-empirical)
│   ├── redteam/
│   │   └── prompt_injection.py    # 5-variant adversarial injection test
│   ├── data/                      # benchmark.json (generated)
│   └── results/                   # per-run JSON results (generated)
├── reports/
│   └── DocuLens_Validation_Report.md  # full MRM validation write-up
├── docker/Dockerfile
├── docker-compose.yml
└── requirements.txt
```
