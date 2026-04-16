# MDM_RAG

Advanced RAG + Multi-Agent pipeline for transforming unstructured product documents into validated, PIM-ready structured data.

## Objective
Build an end-to-end system that processes ~5,000 product documents and produces structured attribute records with retrieval traceability, validation, and human approval before export.

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

## Week 1 Commands
- Sample run:
  - `python -m pipeline.run_pipeline --profile sample`
- Full run:
  - `python -m pipeline.run_pipeline --profile full`

Outputs:
- `reports/week1_dataset_profile.json`
- `reports/week1_dataset_profile.md`
- `reports/week1_query_results.json`
- `data/processed/baseline.index`
- `data/processed/baseline_metadata.json`
- `data/processed/baseline_manifest.json`
- `data/processed/week1_metrics_<profile>.json`
