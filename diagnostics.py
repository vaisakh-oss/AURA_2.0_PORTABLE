import datetime
from enum import Enum
import io
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from PIL import Image


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class HealthCheckResult:
    """Represents the outcome of an individual component health check."""

    def __init__(
        self,
        component: str,
        status: HealthStatus,
        check_name: str,
        message: str,
        latency_ms: float,
        details: Optional[Dict[str, Any]] = None,
        checked_at: Optional[str] = None
    ):
        self.component = component
        self.status = status if isinstance(status, HealthStatus) else HealthStatus(status)
        self.check_name = check_name
        self.message = message
        self.latency_ms = round(latency_ms, 2)
        self.details = details or {}
        self.checked_at = checked_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "check_name": self.check_name,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at
        }


class HealthReport:
    """Aggregate report containing overall health status and component checks."""

    def __init__(
        self,
        overall_status: HealthStatus,
        checks: List[HealthCheckResult],
        recommendations: Optional[List[str]] = None,
        checked_at: Optional[str] = None
    ):
        self.overall_status = overall_status if isinstance(overall_status, HealthStatus) else HealthStatus(overall_status)
        self.checks = checks
        self.checked_at = checked_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.healthy_count = sum(1 for c in checks if c.status == HealthStatus.HEALTHY)
        self.degraded_count = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)
        self.critical_count = sum(1 for c in checks if c.status == HealthStatus.CRITICAL)
        self.unknown_count = sum(1 for c in checks if c.status == HealthStatus.UNKNOWN)

        self.recommendations = recommendations or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "checked_at": self.checked_at,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "critical_count": self.critical_count,
            "unknown_count": self.unknown_count,
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations
        }


# =====================================================================
# INDIVIDUAL COMPONENT HEALTH CHECKS
# =====================================================================

def check_database(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect SQLite database connectivity, execution, and required tables."""
    start = time.monotonic()
    comp = "Database"
    check_name = "SQLite Database Integrity & Schema Check"
    required_tables = {
        "conversations", "messages", "memories", "memory_notes",
        "documents", "document_chunks", "agent_runs", "agent_tool_runs", "agent_experiences"
    }

    try:
        if not os.path.exists(db_path):
            import storage
            import knowledge_store
            storage.initialize_database(db_path)
            knowledge_store.initialize_knowledge_tables(db_path)

        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables_found = {row[0] for row in cursor.fetchall()}
        conn.close()

        missing = required_tables - tables_found
        latency = (time.monotonic() - start) * 1000.0

        if missing:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"Required database tables missing: {sorted(list(missing))}",
                latency_ms=latency,
                details={"tables_found": sorted(list(tables_found)), "missing_tables": sorted(list(missing))}
            )

        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Database accessible, query succeeded, and all required tables exist.",
            latency_ms=latency,
            details={"tables_found_count": len(tables_found), "database_path": db_path}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Database accessibility error: {e}",
            latency_ms=latency
        )


def check_tools() -> HealthCheckResult:
    """Inspect tool registry integrity and expected tool registrations."""
    start = time.monotonic()
    comp = "Tool Registry"
    check_name = "Agent Tool Registry Audit"
    expected_tools = ["web_search", "calculator", "python_exec", "save_note", "read_notes", "knowledge_search"]

    try:
        import tools
        registry = getattr(tools, "TOOL_REGISTRY", None)

        if not isinstance(registry, dict):
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message="Tool registry dictionary missing or invalid.",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        missing = [t for t in expected_tools if t not in registry]
        if missing:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"Missing expected registered tool(s): {missing}",
                latency_ms=(time.monotonic() - start) * 1000.0,
                details={"registered_tools": list(registry.keys()), "missing": missing}
            )

        for name, func in registry.items():
            if not callable(func):
                return HealthCheckResult(
                    component=comp,
                    status=HealthStatus.CRITICAL,
                    check_name=check_name,
                    message=f"Registered tool '{name}' is not callable.",
                    latency_ms=(time.monotonic() - start) * 1000.0
                )

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message=f"All {len(registry)} tools registered, unique, and callable.",
            latency_ms=latency,
            details={"registered_tools": list(registry.keys())}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Tool registry audit failure: {e}",
            latency_ms=latency
        )


def check_planner() -> HealthCheckResult:
    """Inspect task planner import and validation logic."""
    start = time.monotonic()
    comp = "Planner"
    check_name = "Task Planner & Rule Validator"

    try:
        import planner

        # 1. Valid plan test
        valid_plan = {
            "needs_plan": True,
            "goal": "Test goal",
            "steps": [{"id": 1, "description": "Calc", "tool": "calculator", "depends_on": []}]
        }
        ok, msg = planner.validate_plan_dict(valid_plan)
        if not ok:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"Planner failed to validate a valid plan: {msg}",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        # 2. Invalid tool test
        invalid_tool_plan = {
            "needs_plan": True,
            "goal": "Test goal",
            "steps": [{"id": 1, "description": "Bad tool", "tool": "unauthorized_tool", "depends_on": []}]
        }
        ok_bad, _ = planner.validate_plan_dict(invalid_tool_plan)
        if ok_bad:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message="Planner accepted an unauthorized tool name.",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        # 3. Invalid forward dependency test
        invalid_dep_plan = {
            "needs_plan": True,
            "goal": "Test goal",
            "steps": [{"id": 1, "description": "Forward dep", "tool": "calculator", "depends_on": [2]}]
        }
        ok_dep, _ = planner.validate_plan_dict(invalid_dep_plan)
        if ok_dep:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message="Planner accepted an invalid forward dependency.",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Planner module operational and plan validation rules working normally.",
            latency_ms=latency,
            details={"allowed_tools": sorted(list(planner.ALLOWED_TOOLS))}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Planner diagnostic error: {e}",
            latency_ms=latency
        )


def check_agent() -> HealthCheckResult:
    """Inspect AuraAgent core instantiation capability."""
    start = time.monotonic()
    comp = "Agent Core"
    check_name = "AuraAgent Core Instantiation Check"

    try:
        from agent import AuraAgent
        test_agent = AuraAgent(api_key="diagnostic_key_check")
        if not test_agent:
            raise RuntimeError("Agent instantiation returned None.")

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="AuraAgent instantiated successfully with resolved core dependencies.",
            latency_ms=latency,
            details={"model": getattr(test_agent, "model", "unknown")}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Agent core instantiation failed: {e}",
            latency_ms=latency
        )


def check_memory(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect user structured memory storage and retrieval."""
    start = time.monotonic()
    comp = "Memory"
    check_name = "User Structured Memory Check"

    try:
        import storage
        import memory
        storage.initialize_database(db_path)

        memories = storage.list_memories(db_path=db_path)
        retrieved = memory.get_relevant_memories("synthetic query", db_path=db_path)

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="User memory storage accessible and query function executed cleanly.",
            latency_ms=latency,
            details={"stored_memories_count": len(memories), "retrieved_count": len(retrieved)}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Memory storage check error: {e}",
            latency_ms=latency
        )


def check_rag(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect knowledge base tables, embedding dependencies, and retrieval search."""
    start = time.monotonic()
    comp = "RAG"
    check_name = "Knowledge Base & Local RAG Search"

    try:
        import knowledge
        import knowledge_store

        # 1. Check database accessibility and required schema tables
        if not os.path.exists(db_path):
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"RAG database file not found at: {db_path}",
                latency_ms=(time.monotonic() - start) * 1000.0,
                details={"database_path": db_path}
            )

        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables_found = {row[0] for row in cursor.fetchall()}
        conn.close()

        required_rag_tables = {"documents", "document_chunks"}
        missing_tables = required_rag_tables - tables_found
        if missing_tables:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"RAG schema missing required table(s): {sorted(list(missing_tables))}",
                latency_ms=(time.monotonic() - start) * 1000.0,
                details={"missing_tables": sorted(list(missing_tables)), "database_path": db_path}
            )

        # 2. Check local embedding model generation
        try:
            test_vec = knowledge.embed_query("synthetic diagnostic test query")
            if not test_vec or len(test_vec) != 384:
                return HealthCheckResult(
                    component=comp,
                    status=HealthStatus.CRITICAL,
                    check_name=check_name,
                    message=f"RAG embedding model returned invalid vector dimension: {len(test_vec) if test_vec else 0} (expected 384)",
                    latency_ms=(time.monotonic() - start) * 1000.0
                )
        except Exception as emb_err:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"RAG embedding model failure: {emb_err}",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        # 3. Check retrieval search function execution
        try:
            results = knowledge.search_knowledge("synthetic test query", top_k=1, db_path=db_path)
            if not isinstance(results, list):
                return HealthCheckResult(
                    component=comp,
                    status=HealthStatus.CRITICAL,
                    check_name=check_name,
                    message="RAG search returned an invalid non-list result.",
                    latency_ms=(time.monotonic() - start) * 1000.0
                )
        except Exception as search_err:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"RAG retrieval query failure: {search_err}",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        docs = knowledge_store.list_documents(db_path=db_path)
        doc_count = len(docs)
        latency = (time.monotonic() - start) * 1000.0

        # 4. Healthy with zero documents (Cold start)
        if doc_count == 0:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.HEALTHY,
                check_name=check_name,
                message="Knowledge base schema and embedding engine operational (0 documents indexed).",
                latency_ms=latency,
                details={"document_count": 0, "cold_start": True, "embedding_model": "all-MiniLM-L6-v2", "vector_dim": 384}
            )

        # 5. Healthy with documents indexed
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message=f"Knowledge base operational with {doc_count} document(s) indexed.",
            latency_ms=latency,
            details={"document_count": doc_count, "cold_start": False, "test_search_returned": len(results)}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"RAG system check error: {e}",
            latency_ms=latency
        )


def check_vision() -> HealthCheckResult:
    """Inspect multimodal vision pipeline and image validation logic."""
    start = time.monotonic()
    comp = "Vision"
    check_name = "Multimodal Vision Pipeline Audit"

    try:
        import vision

        # 1. Invalid bytes test
        val_invalid = vision.validate_image(b"invalid_corrupted_image_data", "test.png")
        if val_invalid.get("valid") is True:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message="Vision module failed to reject invalid corrupted image bytes.",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        # 2. Valid image test (10x10 PNG)
        buf = io.BytesIO()
        img = Image.new("RGB", (10, 10), color="blue")
        img.save(buf, format="PNG")
        valid_bytes = buf.getvalue()

        val_valid = vision.validate_image(valid_bytes, "test.png")
        if not val_valid.get("valid"):
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"Vision module rejected a valid PNG image: {val_valid.get('error')}",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Vision module operational with valid image processing and strict error handling.",
            latency_ms=latency
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Vision pipeline check error: {e}",
            latency_ms=latency
        )


def check_experience(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect experience memory storage and retrieval search."""
    start = time.monotonic()
    comp = "Experience Memory"
    check_name = "Experience Memory Storage & Ranking Check"

    try:
        import storage
        import experience
        storage.initialize_database(db_path)

        exps = experience.list_experiences(limit=50, db_path=db_path)
        retrieved = experience.retrieve_experiences("synthetic task query", db_path=db_path)

        latency = (time.monotonic() - start) * 1000.0
        if not exps:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.DEGRADED,
                check_name=check_name,
                message="Experience memory table exists and retrieval works, but experience storage is currently empty.",
                latency_ms=latency,
                details={"experience_count": 0}
            )

        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Experience memory operational and search query executed cleanly.",
            latency_ms=latency,
            details={"experience_count": len(exps)}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Experience memory check error: {e}",
            latency_ms=latency
        )


def check_analytics(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect operational analytics telemetry tables and aggregation functions."""
    start = time.monotonic()
    comp = "Analytics"
    check_name = "Operational Telemetry & Health Aggregation"

    try:
        import storage
        import analytics
        storage.initialize_database(db_path)

        usage = analytics.get_usage_summary(db_path=db_path)
        health = analytics.get_agent_health(db_path=db_path)

        latency = (time.monotonic() - start) * 1000.0
        if health["total_tasks"] == 0:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.DEGRADED,
                check_name=check_name,
                message="Analytics telemetry operational but no historical agent runs recorded.",
                latency_ms=latency,
                details={"total_tasks": 0}
            )

        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Analytics telemetry operational and aggregation functions executed cleanly.",
            latency_ms=latency,
            details={"total_tasks": health["total_tasks"], "success_rate_percent": health["success_rate_percent"]}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Analytics check error: {e}",
            latency_ms=latency
        )


def check_configuration() -> HealthCheckResult:
    """Inspect environment configuration without exposing sensitive API keys or credentials."""
    start = time.monotonic()
    comp = "Configuration"
    check_name = "Environment Configuration Audit"

    try:
        import config
        api_key = getattr(config, "GEMINI_API_KEY", None)
        model = getattr(config, "GEMINI_MODEL", "gemini-3.5-flash-lite")

        latency = (time.monotonic() - start) * 1000.0
        # NEVER log or display the actual secret value!
        if not api_key or not api_key.strip():
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.DEGRADED,
                check_name=check_name,
                message="Gemini API key is not configured in environment (GEMINI_API_KEY missing).",
                latency_ms=latency,
                details={"gemini_api_key_configured": False, "model": model}
            )

        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Configuration loaded and Gemini API key is configured.",
            latency_ms=latency,
            details={"gemini_api_key_configured": True, "model": model}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Configuration load error: {e}",
            latency_ms=latency
        )


def check_evaluation() -> HealthCheckResult:
    """Inspect evaluation framework cases and evaluator harness initialization (without running full benchmark)."""
    start = time.monotonic()
    comp = "Evaluation Framework"
    check_name = "Evaluation Benchmark Harness Initialization"

    try:
        import evaluation
        import evaluation_cases

        cases = evaluation_cases.get_all_evaluation_cases()
        if len(cases) != 50:
            return HealthCheckResult(
                component=comp,
                status=HealthStatus.CRITICAL,
                check_name=check_name,
                message=f"Evaluation cases count mismatch: expected 50, found {len(cases)}.",
                latency_ms=(time.monotonic() - start) * 1000.0
            )

        harness = evaluation.BenchmarkHarness()
        if not harness:
            raise RuntimeError("Benchmark harness failed to initialize.")

        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.HEALTHY,
            check_name=check_name,
            message="Evaluation harness initialized with 50 deterministic benchmark cases.",
            latency_ms=latency,
            details={"total_cases": len(cases)}
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        return HealthCheckResult(
            component=comp,
            status=HealthStatus.CRITICAL,
            check_name=check_name,
            message=f"Evaluation harness check error: {e}",
            latency_ms=latency
        )


# =====================================================================
# OVERALL HEALTH AGGREGATOR & RECOMMENDATIONS
# =====================================================================

def calculate_overall_health(checks: List[HealthCheckResult]) -> HealthStatus:
    """
    Deterministic overall health aggregation algorithm:
    1. If any component is CRITICAL -> overall = CRITICAL
    2. Else if any component is DEGRADED -> overall = DEGRADED
    3. Else if any component is HEALTHY -> overall = HEALTHY
    4. Otherwise -> UNKNOWN
    """
    statuses = [c.status for c in checks]
    if any(s == HealthStatus.CRITICAL for s in statuses):
        return HealthStatus.CRITICAL
    elif any(s == HealthStatus.DEGRADED for s in statuses):
        return HealthStatus.DEGRADED
    elif any(s == HealthStatus.HEALTHY for s in statuses):
        return HealthStatus.HEALTHY
    return HealthStatus.UNKNOWN


def generate_recommendation(result: HealthCheckResult) -> Optional[str]:
    """Generate deterministic, non-LLM actionable recommendations for degraded or critical checks."""
    if result.status == HealthStatus.HEALTHY:
        return None

    comp = result.component.lower()
    msg = result.message.lower()

    if comp == "database":
        return "Verify that the SQLite database file path is accessible and write permissions are granted."
    elif comp == "configuration":
        return "Configure the required GEMINI_API_KEY environment variable in your .env file."
    elif comp == "tool registry":
        return "Ensure all required tools (web_search, calculator, python_exec, save_note, read_notes, knowledge_search) are registered in tools.py."
    elif comp == "planner":
        return "Check planner.py for syntax errors or invalid plan validation rules."
    elif comp == "agent core":
        return "Verify AuraAgent dependencies and client initialization settings."
    elif comp == "memory":
        return "Reinitialize user memory tables in SQLite if missing."
    elif comp == "rag":
        if "empty" in msg:
            return "Upload documents to populate the RAG knowledge base."
        return "Check that embedding model dependencies (scikit-learn) are installed and knowledge tables exist."
    elif comp == "vision":
        return "Verify PIL (Pillow) image processing libraries are installed."
    elif comp == "experience memory":
        if "empty" in msg:
            return "Run tasks to populate the Experience Memory storage."
        return "Reinitialize agent_experiences table in SQLite if missing."
    elif comp == "analytics":
        if "no historical" in msg or "empty" in msg:
            return "Execute agent tasks to record operational telemetry."
        return "Reinitialize analytics tables (agent_runs, agent_tool_runs) in SQLite."
    elif comp == "evaluation framework":
        return "Ensure evaluation.py and evaluation_cases.py are intact with 50 benchmark cases."

    return f"Inspect component '{result.component}' details and logs for specific diagnostics."


# =====================================================================
# SYSTEM DIAGNOSTIC RUNNERS
# =====================================================================

QUICK_CHECKS = [
    check_database,
    check_tools,
    check_configuration,
    check_agent,
    check_analytics
]

FULL_CHECKS = [
    check_database,
    check_tools,
    check_configuration,
    check_agent,
    check_analytics,
    check_planner,
    check_memory,
    check_rag,
    check_vision,
    check_experience,
    check_evaluation
]


def _execute_check_list(check_funcs: List[Any], db_path: str = "aura.db") -> HealthReport:
    """Execute a list of health check functions safely with failure isolation."""
    results: List[HealthCheckResult] = []

    for check_fn in check_funcs:
        try:
            import inspect
            sig = inspect.signature(check_fn)
            if "db_path" in sig.parameters:
                res = check_fn(db_path=db_path)
            else:
                res = check_fn()
            results.append(res)
        except Exception as e:
            fn_name = check_fn.__name__.replace("check_", "").title()
            results.append(HealthCheckResult(
                component=fn_name,
                status=HealthStatus.UNKNOWN,
                check_name=f"Check {check_fn.__name__}",
                message=f"Check failed unexpectedly: {e}",
                latency_ms=0.0
            ))

    overall = calculate_overall_health(results)
    recommendations: List[str] = []

    for res in results:
        rec = generate_recommendation(res)
        if rec and rec not in recommendations:
            recommendations.append(rec)

    return HealthReport(
        overall_status=overall,
        checks=results,
        recommendations=recommendations
    )


def run_quick_diagnostics(db_path: str = "aura.db") -> Dict[str, Any]:
    """Run lightweight core health diagnostics suitable for frequent UI rendering."""
    report = _execute_check_list(QUICK_CHECKS, db_path=db_path)
    return report.to_dict()


def run_system_diagnostics(db_path: str = "aura.db") -> Dict[str, Any]:
    """Run comprehensive system health diagnostics across all AURA components."""
    report = _execute_check_list(FULL_CHECKS, db_path=db_path)
    return report.to_dict()


# =====================================================================
# AGENT RELIABILITY & FAILURE OBSERVABILITY DIAGNOSTICS
# =====================================================================

def get_agent_reliability_diagnostics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate an operational reliability and failure diagnostics summary from SQLite telemetry.
    Distinguishes configuration errors, provider failures, timeouts, tool failures, and retrieval outcomes.
    """
    target_db = db_path if db_path is not None else (os.environ.get("AURA_DB_PATH") or "aura.db")
    if not os.path.exists(target_db):
        return {
            "status": HealthStatus.UNKNOWN.value,
            "total_runs": 0,
            "message": f"Database file '{target_db}' does not exist."
        }

    try:
        with sqlite3.connect(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_runs';")
            if not cursor.fetchone():
                return {"status": HealthStatus.UNKNOWN.value, "total_runs": 0, "message": "agent_runs table does not exist."}

            cursor.execute("SELECT count(*), final_status FROM agent_runs GROUP BY final_status;")
            status_counts = {row[1]: row[0] for row in cursor.fetchall()}
            total_runs = sum(status_counts.values())

            completed = status_counts.get("completed", 0)
            completed_warning = status_counts.get("completed_with_warning", 0)
            failed = status_counts.get("failed", 0)
            timed_out = status_counts.get("timeout", 0)
            cancelled = status_counts.get("cancelled", 0)
            max_steps = status_counts.get("max_steps_reached", 0)

            # Tool run statistics
            cursor.execute("""
                SELECT tool, count(*), sum(success), sum(retry)
                FROM agent_tool_runs
                GROUP BY tool;
            """)
            tool_stats = {}
            for row in cursor.fetchall():
                t_name, t_count, t_succ, t_ret = row[0], row[1], row[2] or 0, row[3] or 0
                tool_stats[t_name] = {
                    "total_calls": t_count,
                    "success_count": t_succ,
                    "failure_count": t_count - t_succ,
                    "retry_count": t_ret,
                    "success_rate": round(t_succ / t_count, 4) if t_count > 0 else 1.0
                }

            success_count = completed + completed_warning
            success_rate = round(success_count / total_runs, 4) if total_runs > 0 else 1.0

            if total_runs == 0:
                health = HealthStatus.HEALTHY
            elif success_rate >= 0.90:
                health = HealthStatus.HEALTHY
            elif success_rate >= 0.70:
                health = HealthStatus.DEGRADED
            else:
                health = HealthStatus.CRITICAL

            return {
                "status": health.value,
                "total_runs": total_runs,
                "completed": completed,
                "completed_with_warning": completed_warning,
                "failed": failed,
                "timeout": timed_out,
                "cancelled": cancelled,
                "max_steps_reached": max_steps,
                "success_rate": success_rate,
                "status_breakdown": status_counts,
                "tool_statistics": tool_stats,
                "summary": (
                    f"Total Runs: {total_runs} | Success Rate: {success_rate * 100:.1f}% | "
                    f"Completed: {completed}, Warnings: {completed_warning}, Failed: {failed}, "
                    f"Timeouts: {timed_out}, Cancelled: {cancelled}, Max Steps: {max_steps}"
                )
            }
    except Exception as e:
        return {
            "status": HealthStatus.UNKNOWN.value,
            "total_runs": 0,
            "error": str(e),
            "message": f"Failed to query reliability diagnostics: {e}"
        }


def check_agent_reliability(db_path: str = "aura.db") -> HealthCheckResult:
    """Inspect agent execution reliability and failure rates from recorded telemetry."""
    start = time.monotonic()
    comp = "Agent Reliability"
    check_name = "Agent Execution & Failure Observability Check"

    res = get_agent_reliability_diagnostics(db_path=db_path)
    status_str = res.get("status", "UNKNOWN")
    status = HealthStatus(status_str) if status_str in HealthStatus._value2member_map_ else HealthStatus.UNKNOWN
    latency = (time.monotonic() - start) * 1000.0

    msg = res.get("summary") or res.get("message") or "Reliability metrics checked."
    return HealthCheckResult(
        component=comp,
        status=status,
        check_name=check_name,
        message=msg,
        latency_ms=latency,
        details=res
    )
