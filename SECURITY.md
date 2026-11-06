# AURA Security Threat Model & Hardening Specification

## Executive Summary & Core Security Principle

AURA (Autonomous Reasoning & Utility Assistant) is designed around a fundamental security principle:

> **UNTRUSTED DATA MUST NEVER BECOME AUTHORITY.**

All external data sources—including user prompts, web search results, uploaded documents, OCR/image text, memory records, experience summaries, and tool outputs—are strictly classified as **untrusted reference data**. They provide context for decision-making but can **never** override system instructions, alter tool allowlists, bypass security policies, change execution bounds, or trigger unauthorized system actions.

---

## 1. Trust Classification & Boundaries

### TRUSTED (System Authority)
- **Core Source Code**: `agent.py`, `planner.py`, `tools.py`, `storage.py`, `diagnostics.py`, `experience.py`, `memory.py`, `knowledge.py`, `vision.py`, `config.py`.
- **Tool Dispatcher & Allowlist**: `TOOL_REGISTRY` in `tools.py` and `ALLOWED_TOOLS` in `planner.py`.
- **Schema & Rule Validators**: `validate_plan_dict()`, `_validate_python_ast()`, `validate_image()`, `sanitize_filename()`.
- **Local Security Policies**: AST-based sandbox rules, execution step/retry/replan bounds, path traversal prevention, parameterized SQL queries.

### UNTRUSTED (Reference Data Only)
- **User Prompts**: Direct inputs from chat UI or API calls.
- **Web Search Results**: Snippets and URLs returned by `web_search()`.
- **Uploaded Documents**: Text extracted from PDFs, text files, or markdown documents.
- **Visual Image Content**: Text or structural features extracted from image attachments via `vision.py`.
- **Structured User Memory**: Facts and preferences stored in SQLite `memories` table.
- **Experience Memory**: Historical execution summaries stored in SQLite `agent_experiences` table.
- **Tool Execution Results**: Text output or error strings returned by tools (`python_exec`, `calculator`, `save_note`, etc.).
- **External Model Responses**: Raw output strings from Gemini API.

---

## 2. Asset Inventory & Protection Mechanisms

| Asset Name | Confidentiality | Integrity | Availability | Protection Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **`GEMINI_API_KEY`** | **CRITICAL** | High | High | Environment variable only. Never logged, printed, stored in SQLite DB, exposed in telemetry, or rendered in UI diagnostics. Redacted as `gemini_api_key_configured: True/False`. |
| **Workspace & Filesystem** | High | **CRITICAL** | High | Path traversal prevention (`Path(filename).name`, character sanitization), safe local scratch directory, 10 MB image/doc size caps. |
| **SQLite Database (`aura.db`)** | High | **CRITICAL** | High | 100% Parameterized SQL (`?`) queries. Schema migrations and foreign key constraints enabled. No raw SQL string interpolation. |
| **User Conversations & Notes** | Medium | High | High | Bounded SQLite storage, HTML escaping in UI views, strict user isolation. |
| **Tool Execution Engine** | High | **CRITICAL** | **CRITICAL** | Fixed `ALLOWED_TOOLS` allowlist, AST-based Python executor pre-validation blocking dangerous built-ins/imports/attributes, 5s timeout. |
| **Experience & RAG Memory** | Medium | High | High | Reference DATA ONLY bounded in `<memory_context>` / `<experience_memory>` tags with prompt neutralization rules. |

---

## 3. Trust Boundary Analysis

```
User Prompt (UNTRUSTED)
   ↓ [Boundary A: Schema & Intent Validation]
Planner (TRUSTED)
   ↓ [Boundary B: Allowlist Enforcement]
Tools (TRUSTED Dispatcher → UNTRUSTED Results)
   ↓ [Boundary C: AST & Path Traversal Guards]
External Data / Docs / Images (UNTRUSTED)
   ↓ [Boundary D: Prompt XML Bounding & Redaction]
Agent Reasoning & ReAct Execution (TRUSTED State Machine)
   ↓ [Boundary E: Parameterized SQL Writes]
Persistence / Analytics / Diagnostics (TRUSTED Engine)
```

### Boundary Checks & Guards:
- **Boundary A (User → Planner)**: User goals cannot add custom tools or change system rules. Evaluated via `planner.validate_plan_dict()`.
- **Boundary B (Planner → Tools)**: Tool execution is strictly dispatched via `TOOL_REGISTRY`. Unknown tool names are rejected immediately.
- **Boundary C (Tools → Subprocess/Filesystem)**: `python_exec` uses AST pre-validation (`_validate_python_ast`) blocking `os`, `sys`, `subprocess`, `open()`, `eval()`, `exec()`, and introspection attributes (`__subclasses__`, `__globals__`).
- **Boundary D (External Data → Agent)**: Documents, web search, memory, and OCR text are wrapped in explicit reference XML blocks (`<memory_context>`, `<experience_memory>`, `<knowledge_context>`) instructing the model that contents cannot override system rules.
- **Boundary E (Agent → Database)**: All database interactions use parameterized queries (`cursor.execute(sql, params)`).

---

## 4. Authority Hierarchy

To prevent lower-layer prompt injection or instruction hijacking, AURA enforces a strict 7-tier authority hierarchy:

1. **System & Security Policy** *(Highest Authority: Hard-coded security rules, AST guards, step bounds)*
2. **Application & Tool Policy** *(Allowed tool registry, path traversal guards, size limits)*
3. **Validated Execution Plan** *(Validated multi-step DAG structure)*
4. **User Request** *(Explicit goal submitted by user)*
5. **Structured User Memory** *(Reference facts and preferences)*
6. **Knowledge & Document Context** *(Reference RAG search results)*
7. **Tool / External Results** *(Reference tool outputs and web search snippets)*

*Rule: Lower-tier data can NEVER override higher-tier authority.*

---

## 5. Residual Risks & Defensive Boundaries

1. **Model Non-Determinism**: LLM reasoning may misinterpret instructions despite prompt boundaries. Mitigated by hard-coded Python-level AST validators and plan schema checkers.
2. **Third-Party Libraries**: Underlying Python libraries (`pypdf`, `Pillow`, `google-genai`) could contain unpatched CVEs. Mitigated by strict file size limits (10 MB) and image format verification (`PIL.Image.verify()`).
3. **Local Scope**: Security model assumes local environment isolation. API key protection relies on proper host `.env` file permissions.

*Disclaimer: This document specifies tested local defenses and threat boundaries. It does not constitute an absolute guarantee against all zero-day vulnerabilities.*
