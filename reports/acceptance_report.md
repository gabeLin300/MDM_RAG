# Acceptance Report

- Profile: sample
- Input: data/raw/100_sample_advanced_rag.csv
- Parsed documents: 100
- Chunks created: 1252
- Index size: 1252
- Embedding backend: sentence-transformers
- Index backend: faiss
- Hybrid sparse enabled: True
- Reranker enabled: True
- Query latency p50 (ms): 706.91
- Query latency p95 (ms): 764.78
- Orchestrator backend: sentence-transformers
- Products processed: 100
- Products failed: 0

## Deliverable Checklist
- [x] Ingestion pipeline evidence generated
- [x] Vector store populated
- [x] Hybrid retrieval outputs generated
- [x] Manifest and metrics reports generated
- [x] Full vector store indexed with all documents
- [x] Hybrid search and reranking enabled
- [x] Batch orchestrator processed all products
- [x] Extracted attributes saved to: data\processed\extracted_attributes_sample.json
- [x] PIM JSON export saved to: data\processed\pim_exports\pim_export_sample.json
- [x] PIM CSV export saved to: data\processed\pim_exports\pim_export_sample.csv