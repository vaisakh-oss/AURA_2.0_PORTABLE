# MIRA 2.0 — Agentic Multimodal RAG System for Intelligent Document Understanding

An enterprise-grade AI reasoning assistant that combines a bounded ReAct execution loop, a dynamic tool registry (including a safe math calculator), and a hybrid local RAG pipeline (combining dense vector semantic search using `all-MiniLM-L6-v2` and sparse lexical BM25 matching via Reciprocal Rank Fusion). Designed to provide secure, traceable, and fully grounded document insights with automated citation validation.

## 🚀 Key Engineering Contributions
- **Unified Execution Fabric:** One cohesive framework connecting conversation, multi-step ReAct loops, tool orchestration, and persistent storage.
- **Hybrid Retrieval Engine:** Combines sentence-transformer semantic vector weights with sparse BM25 lexical term matching for precise grounding.
- **Operations & Observability:** Live traceable execution timelines, non-blocking asynchronous APIs, database connection locks, and localized database security authorizers.
