# 🏗️ System Architecture — Advanced RAG + Multi-Agent PIM Extraction

## 📌 High-Level Overview

This system transforms unstructured product documentation (~5,000 files) into structured, validated, and PIM-ready product attributes using an Advanced RAG pipeline combined with a multi-agent extraction system and human-in-the-loop approval workflow.

---

## 🔄 End-to-End Data Flow

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
 Human-in-the-Loop Approval Workflow (simple UI with streamlit)
(review, edit, approve, reject)
        ↓
 PIM Export Layer
(JSON / CSV / API-ready output)