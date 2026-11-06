import datetime
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional
import storage
from storage import DB_FILE, get_db_connection, initialize_database, get_default_db_file
from tools import TOOL_REGISTRY

logger = logging.getLogger("AURA_ANALYTICS")

ALLOWED_TOOLS = set(TOOL_REGISTRY.keys()) if TOOL_REGISTRY else {
    "calculator", "web_search", "python_exec", "save_note", "read_notes", "knowledge_search"
}


def aggregate_run_metrics(
    events: List[Dict[str, Any]],
    result_status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deterministically convert raw safe execution events into operational numerical metrics.
    Guarantees no exposure of prompts, CoT, raw responses, or credentials.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    valid_events = [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []

    if not valid_events:
        status = result_status or "completed"
        is_succ = 1 if status in ("completed", "completed_with_warning") else 0
        is_warn = 1 if status == "completed_with_warning" else 0
        is_err = 1 if status in ("failed", "max_steps_reached") else 0
        return {
            "started_at": now_iso,
            "completed_at": now_iso,
            "duration_ms": 0.0,
            "final_status": status,
            "planned": 0,
            "total_steps": 0,
            "completed_steps": 0,
            "failed_steps": 0,
            "blocked_steps": 0,
            "retry_count": 0,
            "replan_count": 0,
            "verification_status": "skipped",
            "tool_call_count": 0,
            "success": is_succ,
            "warning": is_warn,
            "error": is_err,
            "memory_used": 0,
            "rag_used": 0,
            "vision_used": 0,
            "tool_runs": []
        }

    first_ts = valid_events[0].get("timestamp") or time.time()
    last_ts = valid_events[-1].get("timestamp") or first_ts

    start_iso = datetime.datetime.fromtimestamp(first_ts, tz=datetime.timezone.utc).isoformat()
    completed_iso = datetime.datetime.fromtimestamp(last_ts, tz=datetime.timezone.utc).isoformat()

    elapsed = valid_events[-1].get("elapsed_time")
    if elapsed is not None and isinstance(elapsed, (int, float)) and elapsed > 0:
        duration_ms = round(float(elapsed) * 1000.0, 2)
    else:
        duration_ms = round(max(0.0, (last_ts - first_ts) * 1000.0), 2)


    planned = 0
    total_steps = 0
    completed_steps = 0
    failed_steps = 0
    blocked_steps = 0
    retry_count = 0
    replan_count = 0
    verification_status = "skipped"
    memory_used = 0
    rag_used = 0
    vision_used = 0
    inferred_status = None

    tool_start_map: Dict[str, Dict[str, Any]] = {}
    completed_tool_runs: List[Dict[str, Any]] = []

    for evt in events:
        if not isinstance(evt, dict):
            continue

        e_type = str(evt.get("type", ""))

        if e_type in ("plan_start", "plan_created"):
            planned = 1

        if e_type == "image_start" or e_type == "image_success":
            vision_used = 1

        if e_type == "step_retrying":
            retry_count += 1

        if e_type == "replanning":
            replan_count += 1

        if e_type == "verification_completed":
            if evt.get("is_warning") or not evt.get("success", True):
                verification_status = "warning"
            elif evt.get("success"):
                verification_status = "passed"
            else:
                verification_status = "failed"
        elif e_type == "verification_start" and verification_status == "skipped":
            verification_status = "passed"

        if e_type in ("completed", "completed_with_warning", "failed", "max_steps_reached", "timeout", "cancelled"):
            inferred_status = e_type

        # Tool Tracking & Tool Metrics
        tool_name = evt.get("tool")
        if tool_name:
            tool_str = str(tool_name).strip()
            if tool_str in ("save_note", "read_notes"):
                memory_used = 1
            if tool_str == "knowledge_search":
                rag_used = 1

            if e_type == "tool_start":
                total_steps += 1
                tool_start_map[tool_str] = {
                    "tool": tool_str,
                    "started_at": evt.get("timestamp") or time.time(),
                    "retry": 1 if retry_count > 0 else 0
                }
            elif e_type == "tool_success":
                completed_steps += 1
                start_info = tool_start_map.pop(tool_str, {})
                t_start = start_info.get("started_at", evt.get("timestamp") or time.time())
                t_end = evt.get("timestamp") or time.time()
                t_dur = round(max(0.0, (t_end - t_start) * 1000.0), 2)
                t_start_iso = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc).isoformat()
                t_end_iso = datetime.datetime.fromtimestamp(t_end, tz=datetime.timezone.utc).isoformat()
                completed_tool_runs.append({
                    "tool": tool_str,
                    "started_at": t_start_iso,
                    "completed_at": t_end_iso,
                    "duration_ms": t_dur,
                    "success": 1,
                    "retry": start_info.get("retry", 0),
                    "error_type": None
                })
            elif e_type == "tool_error":
                failed_steps += 1
                start_info = tool_start_map.pop(tool_str, {})
                t_start = start_info.get("started_at", evt.get("timestamp") or time.time())
                t_end = evt.get("timestamp") or time.time()
                t_dur = round(max(0.0, (t_end - t_start) * 1000.0), 2)
                t_start_iso = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc).isoformat()
                t_end_iso = datetime.datetime.fromtimestamp(t_end, tz=datetime.timezone.utc).isoformat()
                completed_tool_runs.append({
                    "tool": tool_str,
                    "started_at": t_start_iso,
                    "completed_at": t_end_iso,
                    "duration_ms": t_dur,
                    "success": 0,
                    "retry": start_info.get("retry", 0),
                    "error_type": "TOOL_ERROR"
                })

    final_status = result_status or inferred_status or "completed"
    is_success = 1 if final_status in ("completed", "completed_with_warning") else 0
    is_warning = 1 if (final_status == "completed_with_warning" or verification_status == "warning" or final_status == "cancelled") else 0
    is_error = 1 if final_status in ("failed", "max_steps_reached", "timeout") else 0

    tool_call_count = len(completed_tool_runs) + len(tool_start_map)

    return {
        "started_at": start_iso,
        "completed_at": completed_iso,
        "duration_ms": duration_ms,
        "final_status": final_status,
        "planned": planned,
        "total_steps": max(total_steps, completed_steps + failed_steps),
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "blocked_steps": blocked_steps,
        "retry_count": retry_count,
        "replan_count": replan_count,
        "verification_status": verification_status,
        "tool_call_count": tool_call_count,
        "success": is_success,
        "warning": is_warning,
        "error": is_error,
        "memory_used": memory_used,
        "rag_used": rag_used,
        "vision_used": vision_used,
        "tool_runs": completed_tool_runs
    }


def record_agent_run(
    events: List[Dict[str, Any]],
    conversation_id: Optional[str] = None,
    message_id: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Persist sanitized agent run operational metrics into SQLite database.
    Rejects sensitive payload data and enforces retention limit (1000 runs).
    """
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    res_status = result.get("status") if isinstance(result, dict) else None
    metrics = aggregate_run_metrics(events, result_status=res_status)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_runs (
                conversation_id, message_id, started_at, completed_at, duration_ms,
                final_status, planned, total_steps, completed_steps, failed_steps,
                blocked_steps, retry_count, replan_count, verification_status,
                tool_call_count, success, warning, error, memory_used, rag_used,
                vision_used, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id, message_id, metrics["started_at"], metrics["completed_at"], metrics["duration_ms"],
                metrics["final_status"], metrics["planned"], metrics["total_steps"], metrics["completed_steps"],
                metrics["failed_steps"], metrics["blocked_steps"], metrics["retry_count"], metrics["replan_count"],
                metrics["verification_status"], metrics["tool_call_count"], metrics["success"], metrics["warning"],
                metrics["error"], metrics["memory_used"], metrics["rag_used"], metrics["vision_used"], created_at
            )
        )
        run_id = cursor.lastrowid

        # Insert Tool Runs
        for tr in metrics.get("tool_runs", []):
            tool_name = str(tr["tool"]).strip()
            if tool_name not in ALLOWED_TOOLS:
                continue
            cursor.execute(
                """
                INSERT INTO agent_tool_runs (
                    run_id, tool, started_at, completed_at, duration_ms, success, retry, error_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, tool_name, tr["started_at"], tr["completed_at"], tr["duration_ms"],
                    tr["success"], tr["retry"], tr.get("error_type")
                )
            )

        conn.commit()

    enforce_retention_policy(limit=1000, db_path=effective_db)
    return run_id


def record_tool_run(
    run_id: int,
    tool: str,
    duration_ms: float,
    success: bool,
    retry: bool = False,
    error_type: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Explicitly record a single tool execution for an existing run."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    tool_name = str(tool).strip()
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Unknown tool '{tool_name}'. Allowed tools: {sorted(list(ALLOWED_TOOLS))}")

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_tool_runs (
                run_id, tool, started_at, completed_at, duration_ms, success, retry, error_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, tool_name, now_iso, now_iso, max(0.0, duration_ms),
                1 if success else 0, 1 if retry else 0, error_type
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_run(run_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve detailed operational metadata for a historical run."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None
        run_dict = dict(row)

        cursor.execute("SELECT * FROM agent_tool_runs WHERE run_id = ? ORDER BY id ASC", (run_id,))
        tool_rows = cursor.fetchall()
        run_dict["tool_runs"] = [dict(tr) for tr in tool_rows]
        return run_dict


def list_recent_runs(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List recent agent operational runs ordered by newest first."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (max(1, limit),))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_tool_statistics(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Calculate aggregate performance statistics for registered tools.
    Includes only authorized tools.
    """
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    stats: Dict[str, Dict[str, Any]] = {
        tool: {"tool": tool, "calls": 0, "success": 0, "failure": 0, "avg_duration_ms": 0.0, "success_rate": 0.0}
        for tool in sorted(list(ALLOWED_TOOLS))
    }

    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tool,
                   COUNT(*) as total_calls,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as total_success,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as total_failure,
                   AVG(duration_ms) as avg_dur
            FROM agent_tool_runs
            GROUP BY tool
        """)
        rows = cursor.fetchall()

        for row in rows:
            t_name = row["tool"]
            if t_name in stats:
                calls = row["total_calls"] or 0
                succ = row["total_success"] or 0
                fail = row["total_failure"] or 0
                avg_d = round(row["avg_dur"] or 0.0, 2)
                rate = round((succ / calls * 100.0), 1) if calls > 0 else 0.0
                stats[t_name] = {
                    "tool": t_name,
                    "calls": calls,
                    "success": succ,
                    "failure": fail,
                    "avg_duration_ms": avg_d,
                    "success_rate": rate
                }

    return [stats[tool] for tool in sorted(list(ALLOWED_TOOLS))]


def get_status_statistics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate counts and percentage breakdown for task outcomes."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM agent_runs")
        total = cursor.fetchone()["total"] or 0

        counts = {
            "completed": 0,
            "completed_with_warning": 0,
            "failed": 0,
            "max_steps_reached": 0,
            "blocked": 0
        }

        if total > 0:
            cursor.execute("""
                SELECT final_status, COUNT(*) as cnt
                FROM agent_runs
                GROUP BY final_status
            """)
            for row in cursor.fetchall():
                st_name = row["final_status"]
                if st_name in counts:
                    counts[st_name] = row["cnt"]
                elif st_name == "blocked":
                    counts["blocked"] = row["cnt"]

        percentages = {}
        for key, val in counts.items():
            percentages[key] = round((val / total * 100.0), 1) if total > 0 else 0.0

        return {
            "total_runs": total,
            "counts": counts,
            "percentages": percentages
        }


def get_average_latency(db_path: Optional[str] = None) -> Dict[str, float]:
    """Calculate average operational latency metrics across run types."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(duration_ms) as avg_total FROM agent_runs")
        avg_total = cursor.fetchone()["avg_total"] or 0.0

        cursor.execute("SELECT AVG(duration_ms) as avg_planned FROM agent_runs WHERE planned = 1")
        avg_planned = cursor.fetchone()["avg_planned"] or 0.0

        cursor.execute("SELECT AVG(duration_ms) as avg_direct FROM agent_runs WHERE planned = 0")
        avg_direct = cursor.fetchone()["avg_direct"] or 0.0

        cursor.execute("SELECT AVG(duration_ms) as avg_tool FROM agent_tool_runs")
        avg_tool = cursor.fetchone()["avg_tool"] or 0.0

        return {
            "overall_avg_ms": round(avg_total, 2),
            "planned_avg_ms": round(avg_planned, 2),
            "direct_avg_ms": round(avg_direct, 2),
            "tool_avg_ms": round(avg_tool, 2)
        }


def get_agent_health(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Produce a deterministic mathematical health scorecard.
    No subjective AI-generated metrics.
    """
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN warning = 1 THEN 1 ELSE 0 END) as warning_count,
                SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as failure_count,
                AVG(duration_ms) as avg_duration,
                AVG(total_steps) as avg_steps,
                AVG(retry_count) as avg_retries,
                SUM(replan_count) as total_replans
            FROM agent_runs
        """)
        row = dict(cursor.fetchone())

        total = row["total"] or 0
        succ = row["success_count"] or 0
        warn = row["warning_count"] or 0
        fail = row["failure_count"] or 0
        avg_dur_ms = row["avg_duration"] or 0.0
        avg_steps = row["avg_steps"] or 0.0
        avg_retries = row["avg_retries"] or 0.0
        total_replans = row["total_replans"] or 0

        succ_rate = round((succ / total * 100.0), 1) if total > 0 else 0.0

        # Determine most used tool
        cursor.execute("""
            SELECT tool, COUNT(*) as cnt
            FROM agent_tool_runs
            GROUP BY tool
            ORDER BY cnt DESC
            LIMIT 1
        """)
        tool_row = cursor.fetchone()
        most_used_tool = tool_row["tool"] if tool_row else "None"

        return {
            "total_tasks": total,
            "success_count": succ,
            "warning_count": warn,
            "failure_count": fail,
            "success_rate_percent": succ_rate,
            "avg_duration_seconds": round(avg_dur_ms / 1000.0, 2),
            "avg_steps": round(avg_steps, 1),
            "avg_retries": round(avg_retries, 2),
            "total_replans": total_replans,
            "most_used_tool": most_used_tool
        }


def get_usage_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate operational feature usage counters across all recorded runs."""
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                SUM(planned) as planning_count,
                SUM(CASE WHEN planned = 1 OR retry_count > 0 OR replan_count > 0 THEN 1 ELSE 0 END) as adaptive_execution_count,
                SUM(memory_used) as memory_count,
                SUM(rag_used) as rag_count,
                SUM(vision_used) as vision_count,
                SUM(replan_count) as total_replans,
                SUM(CASE WHEN verification_status != 'skipped' THEN 1 ELSE 0 END) as verification_count
            FROM agent_runs
        """)
        row = dict(cursor.fetchone())

        # Experience Memory Counters
        cursor.execute("""
            SELECT
                COUNT(*) as created_cnt,
                SUM(use_count) as reused_cnt,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as succ_cnt
            FROM agent_experiences
        """)
        exp_row = cursor.fetchone()
        exp_created = exp_row["created_cnt"] or 0 if exp_row else 0
        exp_reused = exp_row["reused_cnt"] or 0 if exp_row else 0
        exp_succ = exp_row["succ_cnt"] or 0 if exp_row else 0
        exp_succ_rate = round((exp_succ / exp_created * 100.0), 1) if exp_created > 0 else 0.0

        return {
            "planning_usage": row["planning_count"] or 0,
            "adaptive_execution_usage": row["adaptive_execution_count"] or 0,
            "memory_usage": row["memory_count"] or 0,
            "rag_usage": row["rag_count"] or 0,
            "vision_usage": row["vision_count"] or 0,
            "replanning_frequency": row["total_replans"] or 0,
            "verification_frequency": row["verification_count"] or 0,
            "experiences_created": exp_created,
            "experiences_retrieved": exp_reused,
            "experiences_reused": exp_reused,
            "experience_success_rate": exp_succ_rate
        }



def enforce_retention_policy(limit: int = 1000, db_path: Optional[str] = None) -> int:
    """
    Prune agent_runs older than `limit` most recent runs.
    Does NOT touch conversations, messages, memory_notes, memories, or documents.
    """
    effective_db = db_path if db_path is not None else get_default_db_file()
    initialize_database(effective_db)
    if limit <= 0:
        return 0

    with get_db_connection(effective_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM agent_runs
            WHERE id NOT IN (
                SELECT id FROM agent_runs ORDER BY id DESC LIMIT ?
            )
            """,
            (limit,)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count
