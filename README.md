# MDM_RAG

Advanced RAG + Multi-Agent pipeline for transforming unstructured product documents into validated, PIM-ready structured data.

## Objective
Build an end-to-end system that processes ~5,000 product documents and produces structured attribute records with retrieval traceability, validation, and human approval before export.

## Installation
1. Clone the repository and enter the project directory.
   ```bash
   git clone https://github.com/gabeLin300/MDM_RAG
   cd MDM_RAG
   ```
2. Create and activate a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade `pip` and install all project dependencies.
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

## Optional Setup
- If you want to run the multi-agent extraction smoke test, set your Groq API key first:
  ```bash
  export GROQ_API_KEY=your_key_here
  ```
- If Hugging Face model downloads are slow or rate-limited, you can optionally set:
  ```bash
  export HF_TOKEN=your_token_here
  ```

## Architecture

```text
CSV Upload (Raw Dataset)
        ↓
Ingestion Layer
        ↓
Preprocessing Layer
(cleaning, parsing, normalization, language detection)
        ↓
Chunking Engine
(semantic + structure-aware splitting)
        ↓
Embedding Generation
(vector representations of chunks)
        ↓
Vector Store (FAISS)
(storage + indexing + metadata)
        ↓
Advanced RAG Retrieval
(hybrid search + reranking)
        ↓
Multi-Agent Extraction System
(attribute extraction + validation + normalization)
        ↓
Human-in-the-Loop Approval Workflow
(review, edit, approve, reject)
        ↓
PIM Export Layer
(JSON / CSV / API-ready output)
```

## Repository Structure
- `ingestion/`: CSV loading and schema checks
- `preprocessing/`: text cleaning, unicode decoding, parsing, normalization, language handling
- `chunking/`: document chunking and metadata attachment
- `embeddings/`: embedding generation and batching
- `vector_store/`: FAISS indexing and persistence
- `retrieval/`: baseline and advanced retrieval (dense/sparse + reranking)
- `agents/`: specialized extraction/validation/normalization agents
- `workflow_approval/`: human review and approval flow
- `pipeline/`: orchestration and end-to-end run scripts
- `schemas/`: structured product output schema definitions
- `app/`: interface components
- `tests/`: test coverage for key modules
- `planning/`: architecture and implementation planning docs

## Phase Plan
1. Foundation
- Build ingestion, preprocessing, chunking, embeddings, vector indexing, and baseline retrieval.

2. Intelligence Layer
- Add hybrid retrieval, metadata filtering, reranking, and multi-agent extraction.

3. Production Layer
- Add human approval workflow, structured export, and full pipeline execution at dataset scale.

## Success Criteria
- Reliable ingestion and preprocessing for 5,000+ documents
- High-quality retrieval with hybrid RAG
- Consistent structured extraction with validation and normalization
- Human validation for trust and correction
- Export outputs ready for downstream PIM ingestion

## Commands
- Sample run:
  - `python -m pipeline.run_pipeline --profile sample`
- Full run:
  - `python -m pipeline.run_pipeline --profile full`

Outputs:
- `reports/dataset_profile.json`
- `reports/dataset_profile.md`
- `reports/query_results.json`
- `reports/acceptance_report.md`
- `data/processed/baseline.index`
- `data/processed/baseline_metadata.json`
- `data/processed/baseline_manifest.json`
- `data/processed/metrics_<profile>.json`
