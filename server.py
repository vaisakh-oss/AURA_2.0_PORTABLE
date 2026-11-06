import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure aura root is on python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
import storage
import agent
import knowledge
import knowledge_store
import analytics
import diagnostics
import tools

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.initialize_database()
    knowledge_store.initialize_knowledge_tables()
    yield

app = FastAPI(
    title="AURA 2.0 API",
    description="Autonomous Universal Reasoning Agent - Enterprise API Gateway",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global agent instance cache & active runs tracking
_agent_instance: Optional[agent.AuraAgent] = None
_active_conversations: Set[str] = set()

# Dynamic provider state tracking
_provider_health: Dict[str, Any] = {
    "status": "unknown",  # "connected" | "unavailable" | "unauthorized" | "degraded" | "unknown"
    "error": None,
    "last_checked": None,
    "model": config.GEMINI_MODEL
}

def update_provider_health(status: str, error: Optional[str] = None):
    _provider_health["status"] = status
    _provider_health["error"] = error
    _provider_health["last_checked"] = time.time()
    _provider_health["model"] = config.GEMINI_MODEL

def get_agent() -> agent.AuraAgent:
    global _agent_instance
    if _agent_instance is None or _agent_instance.model != config.GEMINI_MODEL:
        _agent_instance = agent.AuraAgent(
            model=config.GEMINI_MODEL,
            max_steps=config.MAX_STEPS
        )
    return _agent_instance


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"

class SendMessageRequest(BaseModel):
    content: str
    conversation_id: Optional[str] = None

class CreateMemoryRequest(BaseModel):
    content: str
    category: str = "FACT"
    importance: float = 0.8
    confidence: float = 0.9

class KnowledgeQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"

class UpdateSettingsRequest(BaseModel):
    model: Optional[str] = None
    max_steps: Optional[int] = None
    rag_mode: Optional[str] = None


# -------------------------------------------------------------
# SYSTEM & HEALTH ENDPOINTS
# -------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    """Return backend readiness, active model, capabilities, and provider status."""
    has_key = bool(config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith("your_"))
    current_provider_status = _provider_health["status"]
    if not has_key:
        current_provider_status = "unauthorized"
    return {
        "status": "online",
        "agent": "AURA 2.0",
        "model": config.GEMINI_MODEL,
        "max_steps": config.MAX_STEPS,
        "has_api_key": has_key,
        "provider_status": current_provider_status,
        "provider_error": _provider_health.get("error"),
        "version": "2.0.0",
        "timestamp": time.time()
    }


@app.get("/api/health")
def get_system_health():
    """Run non-destructive subsystem diagnostics."""
    try:
        report = diagnostics.run_quick_diagnostics()
        has_key = bool(config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith("your_"))
        current_provider_status = _provider_health["status"]
        if not has_key:
            current_provider_status = "unauthorized"

        checks = report.get("checks", [])
        # Update or enrich gemini_provider check with dynamic state
        for check in checks:
            if check.get("name") == "gemini_provider":
                if current_provider_status == "unauthorized":
                    check["status"] = "failed"
                    check["details"] = _provider_health.get("error") or "API key missing or restricted by policy."
                elif current_provider_status == "unavailable":
                    check["status"] = "failed"
                    check["details"] = _provider_health.get("error") or "Provider network or service unavailable."
                elif current_provider_status == "degraded":
                    check["status"] = "warning"
                    check["details"] = _provider_health.get("error") or "Provider rate-limited or degraded."
                elif current_provider_status == "connected":
                    check["status"] = "operational"

        return {
            "status": "degraded" if current_provider_status in ("unauthorized", "unavailable") else report.get("overall_status", "healthy"),
            "provider_status": current_provider_status,
            "provider_error": _provider_health.get("error"),
            "checks": checks,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "checks": []
        }


# -------------------------------------------------------------
# CONVERSATION MANAGEMENT
# -------------------------------------------------------------
@app.get("/api/conversations")
def list_conversations():
    """List all saved conversations."""
    try:
        convs = storage.list_conversations()
        return {"conversations": convs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest):
    """Create a new conversation session."""
    try:
        conv_id = storage.create_conversation(title=req.title or "New Conversation")
        return {"id": conv_id, "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str):
    """Get full message turn history for a conversation."""
    try:
        messages = storage.get_messages(conversation_id)
        # Ensure error messages have visible status steps for Inspector activity tab
        for msg in messages:
            if msg.get("is_error") and not msg.get("status_steps"):
                msg["status_steps"] = [
                    {"type": "error", "label": "Provider execution failure", "is_error": True, "elapsed_time": 0.0}
                ]
        return {
            "id": conversation_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages."""
    try:
        success = storage.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# CHAT & REASONING EXECUTION
# -------------------------------------------------------------
@app.post("/api/chat")
async def chat_interaction(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    is_retry: bool = Form(False)
):
    """
    Execute an autonomous reasoning run with support for text, documents, and images.
    Guarantees concurrency locks to prevent duplicate submissions,
    preserves user messages without duplication on retries, and updates provider health dynamically.
    """
    clean_message = message.strip()
    if not clean_message and not file and not is_retry:
        raise HTTPException(status_code=400, detail="Message prompt or attachment is required")

    # Read and inspect attachment if present
    att_bytes: Optional[bytes] = None
    att_name: Optional[str] = None
    att_size: int = 0
    att_type: Optional[str] = None

    if file:
        att_name = file.filename or "attachment"
        att_bytes = await file.read()
        att_size = len(att_bytes)
        ext = Path(att_name).suffix.lower().lstrip(".")
        if ext in ("docx", "doc"):
            att_type = "docx"
        elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
            att_type = "image"
        elif ext in ("pdf", "txt", "md", "csv"):
            att_type = ext
        else:
            att_type = ext or "file"

    # Ensure conversation exists
    target_conv_id = conversation_id
    if not target_conv_id:
        title_snippet = clean_message[:32] if clean_message else (att_name or "New Analysis")
        target_conv_id = storage.create_conversation(title=title_snippet)

    # Concurrency Lock: Prevent duplicate overlapping submissions for the same conversation
    if target_conv_id in _active_conversations:
        raise HTTPException(
            status_code=409,
            detail="A reasoning run is already active for this conversation. Please wait for it to complete."
        )

    _active_conversations.add(target_conv_id)

    try:
        # If this is a retry and clean_message is empty, retrieve previous user goal
        if is_retry and not clean_message:
            past_msgs = storage.get_messages(target_conv_id)
            user_msgs = [m for m in past_msgs if m.get("role") == "user"]
            if user_msgs:
                clean_message = user_msgs[-1].get("content", "")

        # Persist User Message only if not a retry
        if not is_retry:
            has_att = bool(att_bytes is not None)
            storage.add_message(
                conversation_id=target_conv_id,
                role="user",
                content=clean_message or f"Analyze attached {att_type.upper() if att_type else 'file'}: {att_name}",
                attachment_present=has_att,
                attachment_type=att_type,
                attachment_filename=att_name,
                attachment_size=att_size
            )

        # Prepare input for AuraAgent
        agent_inst = get_agent()
        image_bytes = att_bytes if att_type == "image" else None
        image_filename = att_name if att_type == "image" else None
        doc_bytes = att_bytes if att_type in ("docx", "pdf", "txt", "md", "csv") else None
        doc_filename = att_name if att_type in ("docx", "pdf", "txt", "md", "csv") else None

        # Run agent in background executor to prevent event loop blocking
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent_inst.run(
                    user_goal=clean_message,
                    image_bytes=image_bytes,
                    image_filename=image_filename,
                    document_bytes=doc_bytes,
                    document_filename=doc_filename,
                    conversation_id=target_conv_id
                )
            )
        except Exception as e:
            err_msg = f"Sorry, AURA encountered an execution error: {e}"
            fallback_steps = [{"type": "error", "label": "Execution error", "is_error": True, "elapsed_time": 0.1}]
            update_provider_health("degraded", str(e))
            storage.add_message(
                conversation_id=target_conv_id,
                role="assistant",
                content=err_msg,
                is_error=True,
                status_steps=fallback_steps
            )
            return {
                "conversation_id": target_conv_id,
                "role": "assistant",
                "content": err_msg,
                "status": "failed",
                "steps": 0,
                "status_steps": fallback_steps,
                "citations": [],
                "tool_results": [],
                "provider_status": _provider_health["status"]
            }

        final_answer = result.get("final_answer", "")
        run_status = result.get("status", "completed")
        is_error = (run_status == "failed")
        steps = result.get("steps", 1)
        status_steps = result.get("status_steps", [])
        citations = result.get("citations", [])
        tool_results = result.get("tool_results", [])

        # Dynamic Provider Health Calibration
        if run_status in ("completed", "completed_with_warning"):
            update_provider_health("connected", None)
        elif run_status in ("failed", "degraded"):
            diag = result.get("provider_diagnostic", {})
            cat = diag.get("error_category")
            clean_err = diag.get("sanitized_message") or str(result.get("error", ""))

            if cat in ("policy_restricted", "unauthorized") or any(k in clean_err.lower() for k in ["403", "401", "unauthorized", "not allowed by policy", "forbidden", "permission_denied"]):
                update_provider_health("unauthorized", clean_err)
            elif cat == "quota_exceeded" or any(k in clean_err.lower() for k in ["429", "resource_exhausted", "quota", "rate limit"]):
                update_provider_health("degraded", clean_err)
            elif cat in ("service_unavailable", "timeout", "network_error") or any(k in clean_err.lower() for k in ["503", "unavailable", "timeout", "timed out", "econnrefused", "network"]):
                update_provider_health("unavailable", clean_err)
            else:
                update_provider_health("degraded", clean_err)
        elif run_status == "timeout":
            update_provider_health("unavailable", str(result.get("error", "")))

        # If status_steps is empty on error, supply fallback step
        if is_error and not status_steps:
            status_steps = [
                {"type": "error", "label": "Provider execution error", "is_error": True, "elapsed_time": 0.1}
            ]

        # Persist Assistant Message
        storage.add_message(
            conversation_id=target_conv_id,
            role="assistant",
            content=final_answer,
            steps=steps,
            is_error=is_error,
            status_steps=status_steps
        )

        return {
            "conversation_id": target_conv_id,
            "role": "assistant",
            "content": final_answer,
            "status": run_status,
            "steps": steps,
            "status_steps": status_steps,
            "citations": citations,
            "tool_results": tool_results,
            "provider_status": _provider_health["status"],
            "provider_error": _provider_health.get("error"),
            "provider_diagnostic": result.get("provider_diagnostic")
        }
    finally:
        _active_conversations.discard(target_conv_id)


# -------------------------------------------------------------
# KNOWLEDGE BASE & RAG
# -------------------------------------------------------------
@app.get("/api/knowledge/documents")
def list_knowledge_documents():
    """List all indexed documents in knowledge store."""
    try:
        docs = knowledge_store.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/upload")
async def upload_knowledge_document(file: UploadFile = File(...)):
    """Upload and index a document (DOCX, PDF, TXT, MD, CSV) into vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        doc_info = knowledge.index_document(
            filename=file.filename,
            file_bytes=file_bytes
        )
        return {"success": True, "document": doc_info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document indexing failed: {e}")


@app.delete("/api/knowledge/documents/{document_id}")
def delete_knowledge_document(document_id: str):
    """Delete a document and its chunks from knowledge store."""
    try:
        success = knowledge_store.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/query")
def test_knowledge_query(req: KnowledgeQueryRequest):
    """Execute test search against knowledge store with semantic, lexical, or hybrid RRF."""
    try:
        results = knowledge.search_knowledge(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode
        )
        return {
            "query": req.query,
            "mode": req.mode,
            "top_k": req.top_k,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# MEMORY WORKSPACE
# -------------------------------------------------------------
@app.get("/api/memory")
def list_memories(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """Retrieve structured episodic memories with optional category and search filters."""
    try:
        memories = storage.list_memories(memory_type=category)
        if search:
            q_lower = search.lower()
            memories = [m for m in memories if q_lower in m.get("content", "").lower()]
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory")
def create_memory(req: CreateMemoryRequest):
    """Add a verified memory fact or preference."""
    try:
        mem_id = storage.add_memory(
            content=req.content,
            memory_type=req.category,
            importance=req.importance,
            confidence=req.confidence
        )
        return {"success": True, "id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id: int):
    """Delete a memory item."""
    try:
        success = storage.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": True, "id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# ANALYTICS & TELEMETRY
# -------------------------------------------------------------
@app.get("/api/analytics/metrics")
def get_analytics_metrics():
    """Return genuine SQLite telemetry metrics."""
    try:
        health_metrics = analytics.get_agent_health()
        usage_metrics = analytics.get_usage_summary()
        status_stats = analytics.get_status_statistics()
        tool_stats = analytics.get_tool_statistics()
        avg_latency = analytics.get_average_latency()

        return {
            "metrics": health_metrics,
            "usage": usage_metrics,
            "status_distribution": status_stats,
            "tool_distribution": tool_stats,
            "latency": avg_latency,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/history")
def list_execution_history(limit: int = Query(50, le=200)):
    """Return recent execution run traces."""
    try:
        runs = analytics.list_recent_runs(limit=limit)
        return {"runs": runs, "count": len(runs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
@app.get("/api/settings")
def get_settings():
    """Return runtime configuration."""
    return {
        "model": config.GEMINI_MODEL,
        "max_steps": config.MAX_STEPS,
        "rag_mode": "hybrid",
        "temperature": 0.2,
        "database_path": "aura.db",
        "has_api_key": bool(config.GEMINI_API_KEY)
    }


@app.post("/api/settings")
def update_settings(req: UpdateSettingsRequest):
    """Update non-sensitive runtime parameters and persist to .env."""
    updated = config.update_config(model=req.model, max_steps=req.max_steps)
    if req.model:
        update_provider_health("unknown", None)

    # Reset cached agent so new config applies
    global _agent_instance
    _agent_instance = None

    return {
        "success": True,
        "model": updated["model"],
        "max_steps": updated["max_steps"],
        "provider_status": _provider_health["status"]
    }


# -------------------------------------------------------------
# STATIC FILE SERVING FOR PRODUCTION SPA
# -------------------------------------------------------------
dist_path = CURRENT_DIR / "web" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("AURA_HOST", "127.0.0.1")
    port = int(os.getenv("AURA_PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
