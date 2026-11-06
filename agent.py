import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_STEPS,
    LOGS_DIR,
    PROVIDER_TIMEOUT_SECONDS,
    TOOL_TIMEOUT_SECONDS,
    AGENT_RUN_TIMEOUT_SECONDS,
    MAX_TRANSIENT_RETRIES,
    MAX_TOOL_OUTPUT_CHARS
)
from memory import (
    ShortTermMemory,
    load_memory,
    memory_as_text,
    save_memory_fact,
    get_relevant_memories,
    format_memories_for_prompt,
    extract_candidate_memories,
    persist_extracted_memories
)
from prompts import SYSTEM_PROMPT, build_agent_prompt
from tools import (
    TOOL_REGISTRY,
    run_tool,
    redact_sensitive_text,
    is_tool_error,
    is_tool_idempotent
)
import vision
import planner
from planner import TaskPlan, PlanStep, should_plan, generate_plan, verify_plan_execution
import analytics
import experience
import knowledge
import storage


# =====================================================================
# AGENT RELIABILITY & FAILURE EXCEPTION TYPES
# =====================================================================
class AgentExecutionError(Exception):
    """Base exception for AURA agent execution failures."""
    def __init__(self, message: str, diagnostic: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic or {}


class ProviderTimeoutError(AgentExecutionError):
    """Raised when an external model provider operation times out (HTTP 408 / 504 / deadline exceeded)."""
    pass


class PermanentProviderError(AgentExecutionError):
    """Raised when an unrecoverable model provider failure occurs (e.g. 400/401/403/invalid key/bad request)."""
    pass


class ProviderQuotaError(AgentExecutionError):
    """Raised when model provider quota or rate limit is reached (HTTP 429 / RESOURCE_EXHAUSTED)."""
    pass


class ProviderAuthError(PermanentProviderError):
    """Raised for authentication (HTTP 401) or policy restrictions (HTTP 403)."""
    pass


class ProviderNotFoundError(PermanentProviderError):
    """Raised when the specified model does not exist or has been deprecated (HTTP 404)."""
    pass


class ProviderBadRequestError(PermanentProviderError):
    """Raised when request payload or generation parameters are invalid (HTTP 400)."""
    pass


class ProviderUnavailableError(AgentExecutionError):
    """Raised when upstream model provider service is unavailable (HTTP 500 / 502 / 503)."""
    pass


class ProviderParsingError(AgentExecutionError):
    """Raised when model provider returns an empty or unparseable response."""
    pass


class CancellationError(AgentExecutionError):
    """Raised when an execution is cooperatively cancelled by user or caller."""
    pass


def parse_provider_error(
    error: Any,
    model: str = "",
    duration: float = 0.0,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Extract structured provider diagnostic metadata from any provider exception or error response.
    Never exposes API keys, authorization headers, or private user prompt content.
    Accurately classifies HTTP status codes (400, 401, 403, 404, 408, 429, 500, 502, 503, 504),
    provider error codes, error categories, and generates actionable recovery guidance.
    """
    err_str = str(error)
    sanitized_msg = redact_sensitive_text(err_str)

    # 1. Extract HTTP status code
    http_status: Optional[int] = None
    if hasattr(error, "code") and isinstance(error.code, int):
        http_status = error.code
    elif hasattr(error, "status_code") and isinstance(error.status_code, int):
        http_status = error.status_code
    elif hasattr(error, "response") and hasattr(error.response, "status_code"):
        try:
            http_status = int(error.response.status_code)
        except Exception:
            pass

    if http_status is None:
        status_match = re.search(r"\b(400|401|403|404|408|429|500|502|503|504)\b", err_str)
        if status_match:
            http_status = int(status_match.group(1))

    # 2. Extract provider error code / reason
    provider_code: Optional[str] = None
    if hasattr(error, "status") and error.status:
        provider_code = str(error.status).strip()

    if not provider_code:
        code_match = re.search(
            r"\b(RESOURCE_EXHAUSTED|PERMISSION_DENIED|UNAUTHENTICATED|INVALID_ARGUMENT|NOT_FOUND|UNAVAILABLE|DEADLINE_EXCEEDED|INTERNAL)\b",
            err_str
        )
        if code_match:
            provider_code = code_match.group(1)

    # 3. Extract retry_after or time remaining if present
    retry_after: Optional[str] = None
    retry_match = re.search(r"(?:retry in\s+|retryDelay['\":\s]+)([\d\w\.]+)", err_str, re.IGNORECASE)
    if retry_match:
        retry_after = retry_match.group(1).rstrip(".s").strip()

    # 4. Determine Error Category
    low_msg = sanitized_msg.lower()
    if http_status == 429 or provider_code == "RESOURCE_EXHAUSTED" or any(k in low_msg for k in ["quota", "resource_exhausted", "rate limit"]):
        category = "quota_exceeded"
    elif http_status == 403 or "not allowed by policy" in low_msg or provider_code == "PERMISSION_DENIED" or "permission_denied" in low_msg:
        category = "policy_restricted"
    elif http_status == 401 or provider_code == "UNAUTHENTICATED" or any(k in low_msg for k in ["api_key_invalid", "unauthorized", "unauthenticated", "invalid api key"]):
        category = "unauthorized"
    elif http_status == 404 or provider_code == "NOT_FOUND" or any(k in low_msg for k in ["not found", "no longer available", "does not exist"]):
        category = "model_not_found"
    elif http_status == 400 or provider_code == "INVALID_ARGUMENT" or any(k in low_msg for k in ["bad request", "invalid_argument"]):
        category = "bad_request"
    elif http_status in (408, 504) or provider_code == "DEADLINE_EXCEEDED" or any(k in low_msg for k in ["timeout", "timed out", "deadline exceeded"]):
        category = "timeout"
    elif http_status in (500, 502, 503) or provider_code == "UNAVAILABLE" or any(k in low_msg for k in ["503", "unavailable", "high demand", "overloaded"]):
        category = "service_unavailable"
    elif any(k in low_msg for k in ["connection refused", "dns", "failed to connect", "network"]):
        category = "network_error"
    elif "empty response" in low_msg:
        category = "parsing_error"
    else:
        category = "provider_error"

    # 5. Formulate actionable guidance
    if category == "quota_exceeded":
        delay_phrase = f" Daily quota resets in {retry_after}." if retry_after else ""
        guidance = (
            f"Google Gemini Free Tier daily limit reached for model '{model}' (20 requests/day).{delay_phrase} "
            f"Switch to an alternative model in Settings, or enable billing in Google AI Studio."
        )
    elif category == "policy_restricted":
        guidance = (
            f"Request rejected by policy on generativelanguage.googleapis.com (HTTP 403). "
            f"Verify that Generative Language API is enabled and permitted in Google Cloud Console API key restrictions, "
            f"and check that network egress to generativelanguage.googleapis.com is allowed."
        )
    elif category == "unauthorized":
        guidance = "Authentication failed (HTTP 401). Please check GEMINI_API_KEY in your .env configuration."
    elif category == "model_not_found":
        guidance = f"Model '{model}' is not available or has been deprecated on the Gemini API. Please select an active model in Settings."
    elif category == "bad_request":
        guidance = "Invalid request arguments sent to model. Please check prompt parameters."
    elif category == "service_unavailable":
        guidance = f"Google Gemini model '{model}' is experiencing high temporary demand (HTTP 503). Please retry in a few moments."
    elif category == "timeout":
        guidance = f"Provider request timed out ({model}). Consider shortening prompt context or retrying."
    else:
        guidance = f"Gemini provider error ({category}). Please verify model availability and network connection."

    return {
        "provider": "gemini",
        "model": model,
        "endpoint_host": "generativelanguage.googleapis.com",
        "http_status": http_status,
        "provider_error_code": provider_code,
        "sanitized_message": sanitized_msg[:500],
        "duration_seconds": round(duration, 3),
        "retry_count": attempt,
        "error_category": category,
        "has_response_body": hasattr(error, "response") and bool(getattr(error, "response", None)),
        "retry_after": retry_after,
        "actionable_guidance": guidance
    }



# Setup logging to log operational events safely without exposing secrets
logger = logging.getLogger("AURA_AGENT")
logger.setLevel(logging.INFO)

log_file_path = LOGS_DIR / "agent.log"
if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )
    logger.addHandler(stream_handler)


TOOL_LABEL_MAP = {
    "web_search": "Searching the web",
    "calculator": "Calculating",
    "python_exec": "Running analysis",
    "save_note": "Saving to memory",
    "read_notes": "Reading memory",
    "knowledge_search": "Searching knowledge",
}


def get_tool_label(tool_name: str) -> str:
    """Return human-friendly operational label for a tool."""
    return TOOL_LABEL_MAP.get(str(tool_name).strip(), "Running tool")


SENSITIVE_PATTERNS = [
    # Google API Key (AI Studio & Vertex Express keys)
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"AQ\.[A-Za-z0-9_\-\.]{20,}"), "[REDACTED_API_KEY]"),
    # OpenAI / generic bearer secret keys
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    # GitHub personal access tokens
    (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}"), "[REDACTED_TOKEN]"),
    # HTTP Bearer tokens
    (re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}"), "Bearer [REDACTED_TOKEN]"),
    # Password / secret / token assignments (e.g. password=xyz, "api_key": "xyz")
    (re.compile(r"(?i)\b(api[_-]?key|password|passwd|secret|token|auth[_-]?token|access[_-]?token)\b\s*[:=]\s*['\"]?([^\s'\",]{4,})['\"]?"), r"\1=[REDACTED_SECRET]"),
    # High entropy 32-64 hex strings labeled as key/secret/token
    (re.compile(r"(?i)\b(key|secret|token)\s*[:=]\s*[a-f0-9]{32,64}\b"), r"\1=[REDACTED_HEX_KEY]"),
]


def redact_sensitive_text(text: Any) -> str:
    """
    Scrub common API keys, tokens, passwords, and credentials from string representations
    before writing to logs or telemetry.
    """
    if text is None:
        return ""
    val = str(text)
    for pattern, replacement in SENSITIVE_PATTERNS:
        val = pattern.sub(replacement, val)
    return val


def normalize_tool_action_key(tool_name: str, action_input: Any) -> Tuple[str, str]:
    """
    Produce a canonical tuple key (tool_name, normalized_args) for duplicate tool detection.
    Normalizes dictionary keys, JSON strings, and whitespace so identical semantics produce
    identical keys, while distinct arguments remain strictly separate.
    """
    clean_tool = str(tool_name).strip()
    if isinstance(action_input, dict):
        try:
            norm_args = json.dumps(action_input, sort_keys=True)
        except Exception:
            norm_args = str(action_input).strip()
    elif isinstance(action_input, list):
        try:
            norm_args = json.dumps(action_input)
        except Exception:
            norm_args = str(action_input).strip()
    elif isinstance(action_input, str):
        stripped = action_input.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    norm_args = json.dumps(parsed, sort_keys=True)
                elif isinstance(parsed, list):
                    norm_args = json.dumps(parsed)
                else:
                    norm_args = stripped
            except Exception:
                norm_args = stripped
        else:
            norm_args = stripped
    else:
        norm_args = str(action_input).strip() if action_input is not None else ""

    return (clean_tool, norm_args)


def has_explicit_memory_consent(goal: str) -> bool:
    """Check if the user explicitly requested or consented to saving persistent memory."""
    lower = (goal or "").lower().strip()
    return any(p in lower for p in [
        "remember that", "save this to memory", "remember this",
        "save note", "make a note", "save memory", "store this fact",
        "save to memory", "remember:", "note that"
    ])


class AuraAgent:
    """
    Core ReAct Agent for AURA (Autonomous Universal Reasoning Agent).
    Orchestrates memory loading, prompt construction, Gemini model decisions,
    multimodal vision inspection, tool execution, and observation feedback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        event_callback: Optional[Any] = None,
        db_path: Optional[str] = None,
        provider_timeout: Optional[float] = None,
        run_timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.max_steps = max_steps or MAX_STEPS
        self.event_callback = event_callback
        self.db_path = db_path
        self.provider_timeout = float(provider_timeout or PROVIDER_TIMEOUT_SECONDS)
        self.run_timeout = float(run_timeout or AGENT_RUN_TIMEOUT_SECONDS)
        self.max_retries = int(max_retries if max_retries is not None else MAX_TRANSIENT_RETRIES)

        if not self.api_key:
            logger.warning("AuraAgent initialized without GEMINI_API_KEY.")
            self.client = None
        else:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": int(self.provider_timeout * 1000)}
            )

        self.short_term_memory = ShortTermMemory()

    def _emit_event(
        self,
        callback: Optional[Any],
        event_data: Dict[str, Any],
        events_collector: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Emit safe operational event payload without exposing private chain-of-thought,
        internal prompts, API keys, or raw model outputs. Collects event metrics for analytics.
        """
        safe_payload = {
            "type": str(event_data.get("type", "")),
            "label": str(event_data.get("label", "")),
            "timestamp": event_data.get("timestamp") or time.time(),
            "elapsed_time": round(float(event_data.get("elapsed_time", 0.0)), 2)
        }

        if "tool" in event_data:
            safe_payload["tool"] = str(event_data["tool"])
        if "success" in event_data:
            safe_payload["success"] = bool(event_data["success"])
        if "is_error" in event_data:
            safe_payload["is_error"] = bool(event_data["is_error"])
        if "is_warning" in event_data:
            safe_payload["is_warning"] = bool(event_data["is_warning"])

        if events_collector is not None and isinstance(events_collector, list):
            events_collector.append(safe_payload)

        cb = callback or self.event_callback
        if cb and callable(cb):
            # Forward milestone/lifecycle events to external callbacks to preserve backward compatibility
            excluded_from_classic_callback = (
                "context_assembled", "provider_start", "provider_success",
                "persistence_success", "persistence_failed",
                "subtask_started", "tool_selected", "tool_executed",
                "result_validated", "subtask_completed", "recovery_attempted"
            )
            if safe_payload["type"] not in excluded_from_classic_callback:
                try:
                    cb(safe_payload)
                except Exception as cb_err:
                    logger.warning(f"Event callback execution error: {cb_err}")


    def _clean_json_response(self, text: str) -> str:
        """
        Strip markdown code block fences (```json or ```) and leading/trailing whitespace.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_and_validate_response(self, text: str) -> Dict[str, Any]:
        """
        Parse raw model response into JSON and validate against the required schemas.
        Rejects malformed JSON, missing fields, or unauthorized tool names.
        """
        cleaned_text = self._clean_json_response(text)

        try:
            data = json.loads(cleaned_text)
        except Exception as e:
            raise ValueError(f"Malformed JSON output from model: {e}")

        if not isinstance(data, dict):
            raise ValueError("Model output must be a JSON object.")

        if "final_answer" in data:
            final_ans = data["final_answer"]
            if not isinstance(final_ans, str):
                raise ValueError("Field 'final_answer' must be a string.")
            return {
                "type": "final_answer",
                "thought": str(data.get("thought", "")),
                "final_answer": final_ans
            }

        elif "action" in data:
            action_name = data["action"]
            if not isinstance(action_name, str):
                raise ValueError("Field 'action' must be a string.")

            action_name = action_name.strip()
            if action_name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown action '{action_name}'. Allowed actions: {list(TOOL_REGISTRY.keys())}")

            if "action_input" not in data:
                raise ValueError("Missing required field 'action_input' for tool action.")

            action_input = data["action_input"]
            if isinstance(action_input, (dict, list)):
                action_input_str = json.dumps(action_input)
            else:
                action_input_str = str(action_input)

            return {
                "type": "action",
                "thought": str(data.get("thought", "")),
                "action": action_name,
                "action_input": action_input_str
            }
        else:
            raise ValueError("JSON response must contain either 'action' or 'final_answer'.")

        return {}

    def _call_gemini_with_retry(
        self,
        prompt: str,
        image: Optional[Any] = None,
        max_retries: Optional[int] = None
    ) -> str:
        """
        Call Gemini generate_content with structured diagnostic classification,
        bounded transient retries, and immediate fail-fast on permanent errors or daily quota exhaustion.
        Supports text-only and text + PIL Image multimodal requests.
        """
        if not self.client:
            diag = {
                "provider": "gemini",
                "model": self.model,
                "endpoint_host": "generativelanguage.googleapis.com",
                "http_status": 401,
                "provider_error_code": "UNAUTHENTICATED",
                "sanitized_message": "Gemini Client is not initialized. Please set GEMINI_API_KEY in .env.",
                "duration_seconds": 0.0,
                "retry_count": 0,
                "error_category": "unauthorized",
                "has_response_body": False,
                "retry_after": None,
                "actionable_guidance": "Please set GEMINI_API_KEY in .env."
            }
            raise PermanentProviderError("Gemini Client is not initialized. Please set GEMINI_API_KEY in .env.", diagnostic=diag)

        retries_limit = max_retries if max_retries is not None else self.max_retries
        base_delay = 1.5
        contents = [image, prompt] if image is not None else prompt

        for attempt in range(1, retries_limit + 1):
            call_start = time.monotonic()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents
                )
                text_out = None
                if response:
                    try:
                        text_out = response.text
                    except Exception:
                        pass
                    if not text_out and hasattr(response, "candidates") and response.candidates:
                        c = response.candidates[0]
                        if hasattr(c, "content") and hasattr(c.content, "parts"):
                            parts_texts = [p.text for p in c.content.parts if getattr(p, "text", None)]
                            if parts_texts:
                                text_out = "".join(parts_texts)

                if not text_out or not text_out.strip():
                    if attempt < retries_limit:
                        time.sleep(base_delay * attempt)
                        continue
                    dur = time.monotonic() - call_start
                    diag = {
                        "provider": "gemini",
                        "model": self.model,
                        "endpoint_host": "generativelanguage.googleapis.com",
                        "http_status": 200,
                        "provider_error_code": "EMPTY_RESPONSE",
                        "sanitized_message": "Gemini returned an empty response.",
                        "duration_seconds": round(dur, 3),
                        "retry_count": attempt,
                        "error_category": "parsing_error",
                        "has_response_body": bool(response),
                        "retry_after": None,
                        "actionable_guidance": "Model returned an empty response. Please retry with a revised prompt."
                    }
                    raise ProviderParsingError("Gemini returned an empty response.", diagnostic=diag)

                return text_out.strip()

            except AgentExecutionError:
                raise
            except Exception as e:
                dur = time.monotonic() - call_start
                diag = parse_provider_error(e, model=self.model, duration=dur, attempt=attempt)
                category = diag["error_category"]

                # Fail-fast on permanent authorization / authentication / policy errors: 401, 403
                if category in ("policy_restricted", "unauthorized"):
                    logger.error(f"Gemini API authorization failure (HTTP {diag['http_status']}): {diag['sanitized_message']}")
                    raise PermanentProviderError(f"Permanent provider failure: {diag['sanitized_message']}", diagnostic=diag) from e

                # Fail-fast on model not found: 404
                if category == "model_not_found":
                    logger.error(f"Gemini model not found (HTTP 404): {diag['sanitized_message']}")
                    raise PermanentProviderError(f"Model not found: {diag['sanitized_message']}", diagnostic=diag) from e

                # Fail-fast on bad request: 400
                if category == "bad_request":
                    logger.error(f"Gemini bad request (HTTP 400): {diag['sanitized_message']}")
                    raise PermanentProviderError(f"Bad request: {diag['sanitized_message']}", diagnostic=diag) from e

                # For 429 Quota Exhaustion:
                # If it's a daily limit cap with long delay, fail-fast immediately without redundant delays
                if category == "quota_exceeded":
                    retry_after = diag.get("retry_after")
                    is_daily_cap = "daily" in diag["sanitized_message"].lower() or "per day" in diag["sanitized_message"].lower() or (retry_after and any(unit in str(retry_after) for unit in ["h", "d", "m"]))
                    if is_daily_cap or attempt >= retries_limit:
                        logger.error(f"Gemini quota exhausted (HTTP 429): {diag['sanitized_message']}")
                        raise ProviderQuotaError(f"Quota exhausted: {diag['sanitized_message']}", diagnostic=diag) from e

                    delay = min(base_delay ** attempt, 8.0)
                    logger.warning(
                        f"Gemini API transient rate limit (Attempt {attempt}/{retries_limit}): {diag['sanitized_message']}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue

                # Check for timeout errors
                if category == "timeout":
                    if attempt >= retries_limit:
                        logger.error(f"Gemini API timed out after {attempt} attempts: {diag['sanitized_message']}")
                        raise ProviderTimeoutError(f"Gemini API timed out: {diag['sanitized_message']}", diagnostic=diag) from e
                    delay = min(base_delay ** attempt, 8.0)
                    logger.warning(f"Gemini timeout (Attempt {attempt}/{retries_limit}): Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                # Service unavailable / high demand (500/502/503)
                if category in ("service_unavailable", "network_error"):
                    if attempt < retries_limit:
                        delay = min(base_delay ** attempt, 8.0)
                        logger.warning(
                            f"Gemini service transient issue (Attempt {attempt}/{retries_limit}): {diag['sanitized_message']}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Gemini service unavailable after {attempt} attempts: {diag['sanitized_message']}")
                        raise ProviderUnavailableError(f"Gemini API call failed: {diag['sanitized_message']}", diagnostic=diag) from e

                # Default fallback
                if attempt >= retries_limit:
                    logger.error(f"Gemini API call failed on attempt {attempt}/{retries_limit}: {diag['sanitized_message']}")
                    raise AgentExecutionError(f"Gemini API call failed: {diag['sanitized_message']}", diagnostic=diag) from e

                delay = min(base_delay ** attempt, 8.0)
                time.sleep(delay)

        dur = time.monotonic() - call_start
        diag = {
            "provider": "gemini",
            "model": self.model,
            "endpoint_host": "generativelanguage.googleapis.com",
            "http_status": 408,
            "provider_error_code": "DEADLINE_EXCEEDED",
            "sanitized_message": f"Exhausted Gemini API retries ({retries_limit}).",
            "duration_seconds": round(dur, 3),
            "retry_count": retries_limit,
            "error_category": "timeout",
            "has_response_body": False,
            "retry_after": None,
            "actionable_guidance": "Exhausted retries. Check connection and retry."
        }
        raise ProviderTimeoutError(f"Exhausted Gemini API retries ({retries_limit}).", diagnostic=diag)

    def run(
        self,
        user_goal: str,
        image_bytes: Optional[bytes] = None,
        image_filename: Optional[str] = None,
        document_bytes: Optional[bytes] = None,
        document_filename: Optional[str] = None,
        document_text: Optional[str] = None,
        event_callback: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[int] = None,
        cancel_token: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the main ReAct loop for a given user goal, supporting text, documents, and multimodal image input.
        Guarantees consistent terminal states, bounded retries, duplicate execution prevention,
        and telemetry recording for all outcomes (success, warning, failure, timeout, cancelled).
        """
        start_time = time.monotonic()
        deadline = start_time + self.run_timeout
        active_cb = event_callback or self.event_callback
        events_collector: List[Dict[str, Any]] = []
        step = 0

        goal = (user_goal or "").strip()
        target_db = self.db_path or storage.get_default_db_file()

        def emit(evt_data: Dict[str, Any]):
            self._emit_event(active_cb, evt_data, events_collector=events_collector)

        def finish(result_dict: Dict[str, Any]) -> Dict[str, Any]:
            final_st = result_dict.get("status", "completed")
            is_success = (final_st in ("completed", "completed_with_warning"))
            result_dict["success"] = is_success

            try:
                analytics.record_agent_run(
                    events_collector,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    result=result_dict,
                    db_path=target_db
                )
                emit({
                    "type": "persistence_success",
                    "label": "Execution run persisted to database",
                    "success": True,
                    "elapsed_time": time.monotonic() - start_time
                })
            except Exception as record_err:
                logger.warning(f"Failed to record analytics run: {record_err}")
                result_dict["persistence_error"] = str(record_err)
                emit({
                    "type": "persistence_failed",
                    "label": f"Telemetry persistence failed: {record_err}",
                    "is_warning": True,
                    "elapsed_time": time.monotonic() - start_time
                })
                if result_dict.get("status") == "completed":
                    result_dict["status"] = "completed_with_warning"
                    result_dict["persistence_warning"] = (
                        f"Execution completed, but telemetry persistence failed: {record_err}"
                    )

            try:
                ver_st = "passed" if final_st == "completed" else ("warning" if final_st == "completed_with_warning" else "failed")
                plan_obj = result_dict.get("plan")
                plan_steps_desc = [s.get("description", "") for s in plan_obj.get("steps", [])] if isinstance(plan_obj, dict) else []

                tools_used_list = list(dict.fromkeys([
                    str(e["tool"]) for e in events_collector if e.get("type") in ("tool_start", "tool_success") and e.get("tool")
                ]))

                experience.create_experience(
                    goal=goal,
                    tools_used=tools_used_list,
                    plan_steps=plan_steps_desc,
                    final_status=final_st,
                    verification_status=ver_st,
                    total_steps=result_dict.get("steps", 0),
                    retry_count=sum(1 for e in events_collector if e.get("type") == "step_retrying"),
                    replan_count=sum(1 for e in events_collector if e.get("type") == "replanning"),
                    duration_ms=round((time.monotonic() - start_time) * 1000.0, 2),
                    has_image=bool(image_bytes),
                    db_path=target_db
                )

                if retrieved_experiences and final_st in ("completed", "completed_with_warning"):
                    for exp in retrieved_experiences:
                        experience.record_experience_use(exp["id"], db_path=target_db)
            except Exception as exp_err:
                logger.warning(f"Failed to record experience memory: {exp_err}")

            # Ensure safe events and status_steps are attached to result_dict for caller and UI inspection
            result_dict["events"] = list(events_collector)
            result_dict["status_steps"] = [
                {
                    "type": str(ev.get("type", "step")),
                    "label": str(ev.get("label", "")),
                    "is_error": bool(ev.get("is_error", False)),
                    "is_warning": bool(ev.get("is_warning", False)),
                    "elapsed_time": float(ev.get("elapsed_time", 0.0))
                }
                for ev in events_collector
            ]

            return result_dict

        def _check_cancellation():
            if cancel_token is None:
                return
            if callable(cancel_token):
                if cancel_token():
                    raise CancellationError("Execution cancelled by user or caller.")
            elif hasattr(cancel_token, "is_set") and cancel_token.is_set():
                raise CancellationError("Execution cancelled by user or caller.")
            elif hasattr(cancel_token, "is_cancelled") and cancel_token.is_cancelled():
                raise CancellationError("Execution cancelled by user or caller.")
            elif getattr(cancel_token, "cancelled", False):
                raise CancellationError("Execution cancelled by user or caller.")

        def _check_deadline():
            if time.monotonic() > deadline:
                raise ProviderTimeoutError(f"Agent execution exceeded overall run timeout of {self.run_timeout:.1f}s.")

        if not goal:
            res = {
                "status": "failed",
                "error": "User goal cannot be empty.",
                "final_answer": "Error: User goal cannot be empty."
            }
            emit({
                "type": "failed",
                "label": "Error encountered",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            return finish(res)

        # Retrieve relevant past experience memory (max 3)
        retrieved_experiences = experience.retrieve_experiences(goal, has_image=bool(image_bytes), limit=3, db_path=target_db)
        experience_context = experience.build_experience_context(retrieved_experiences) if retrieved_experiences else ""

        run_sources: Dict[int, Dict[str, Any]] = {}
        executed_tool_actions: Dict[Tuple[str, str], str] = {}
        safe_goal = redact_sensitive_text(goal)[:80]
        logger.info(f"Starting AURA Agent run (goal_len={len(goal)}, preview='{safe_goal}', has_image={bool(image_bytes)})")

        emit({
            "type": "agent_start",
            "label": "Understanding request",
            "elapsed_time": time.monotonic() - start_time
        })

        if retrieved_experiences:
            logger.info(f"Retrieved {len(retrieved_experiences)} past experience(s)")

        try:
            _check_cancellation()
            _check_deadline()

            # Auto-detect if image_bytes is actually a DOCX document
            if image_bytes and not document_bytes:
                img_name_lower = (image_filename or "").lower()
                if img_name_lower.endswith((".docx", ".doc")) or (image_bytes.startswith(b"PK\x03\x04") and not img_name_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))):
                    document_bytes = image_bytes
                    document_filename = image_filename or "document.docx"
                    image_bytes = None
                    image_filename = None

            # Validate & preprocess attached image if present
            pil_img = None
            if image_bytes:
                emit({
                    "type": "image_start",
                    "label": "Analyzing image",
                    "elapsed_time": time.monotonic() - start_time
                })
                val_res = vision.validate_image(image_bytes, image_filename or "uploaded_image.png")
                if not val_res["valid"]:
                    err_msg = val_res["error"]
                    emit({
                        "type": "failed",
                        "label": "Image validation failed",
                        "is_error": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    return finish({
                        "status": "failed",
                        "error": err_msg,
                        "final_answer": f"Image validation error: {err_msg}"
                    })
                try:
                    pil_img = vision.preprocess_image(image_bytes)
                    emit({
                        "type": "image_success",
                        "label": "Image analyzed",
                        "success": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                except Exception as pe:
                    emit({
                        "type": "failed",
                        "label": "Image processing failed",
                        "is_error": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    return finish({
                        "status": "failed",
                        "error": str(pe),
                        "final_answer": f"Failed to process image: {pe}"
                    })

            # Validate & extract attached document if present
            doc_context = ""
            if document_bytes or document_text:
                import document_extractor
                doc_fname = document_filename or "document.docx"
                emit({
                    "type": "doc_start",
                    "label": f"Analyzing document {doc_fname}",
                    "elapsed_time": time.monotonic() - start_time
                })
                if document_text:
                    doc_context = document_extractor.wrap_untrusted_document(document_text, filename=doc_fname)
                    emit({
                        "type": "doc_success",
                        "label": f"Document {doc_fname} ready",
                        "success": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                else:
                    try:
                        ext_lower = Path(doc_fname).suffix.lower()
                        if ext_lower == ".docx":
                            extracted_doc_info = document_extractor.extract_docx(document_bytes, filename=doc_fname)
                        else:
                            extracted_doc_info = document_extractor.extract_document(document_bytes, filename=doc_fname)

                        if not extracted_doc_info.get("text", "").strip():
                            emit({
                                "type": "failed",
                                "label": "Document contains no text",
                                "is_error": True,
                                "elapsed_time": time.monotonic() - start_time
                            })
                            return finish({
                                "status": "failed",
                                "error": "No extractable text found in document.",
                                "final_answer": f"Uploaded document '{doc_fname}' contains no readable text."
                            })

                        doc_context = document_extractor.wrap_untrusted_document(extracted_doc_info["text"], filename=doc_fname)
                        char_cnt = extracted_doc_info.get("char_count", len(extracted_doc_info.get("text", "")))
                        emit({
                            "type": "doc_success",
                            "label": f"Document {doc_fname} extracted ({char_cnt} chars)",
                            "success": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                    except Exception as doc_err:
                        err_msg = str(doc_err)
                        emit({
                            "type": "failed",
                            "label": "Document validation failed",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        return finish({
                            "status": "failed",
                            "error": err_msg,
                            "final_answer": f"Document processing error: {err_msg}"
                        })

                if doc_context:
                    run_sources[1] = {"index": 1, "source_label": f"{doc_fname} (Attached Document)"}

            # Retrieve ranked relevant long-term memory context via memory_as_text
            memory_context = memory_as_text(limit=10, query=goal, db_path=target_db)

            # Context Assembly Event
            ctx_summary = []
            if doc_context:
                ctx_summary.append("document context")
            if memory_context:
                ctx_summary.append("memory context")
            if experience_context:
                ctx_summary.append("experience context")
            emit({
                "type": "context_assembled",
                "label": f"Context assembled: {', '.join(ctx_summary) if ctx_summary else 'direct goal'}",
                "success": True,
                "elapsed_time": time.monotonic() - start_time
            })

            # Intelligent Request Understanding & Classification
            from classifier import classify_task, TaskCategory
            task_classification = classify_task(
                goal,
                has_image=bool(image_bytes),
                has_document=bool(doc_context)
            )
            logger.info(
                f"Task Classified: category={task_classification.category.value}, "
                f"needs_plan={task_classification.needs_plan}, "
                f"suggested_tools={task_classification.suggested_tools}"
            )

            if task_classification.is_unsupported:
                emit({
                    "type": "failed",
                    "label": "Unsupported operation rejected",
                    "is_error": True,
                    "elapsed_time": time.monotonic() - start_time
                })
                return finish({
                    "status": "failed",
                    "error": "Unsupported or restricted operation.",
                    "final_answer": "I cannot perform that request. The operation is unsupported or restricted by security policy.",
                    "steps": 0,
                    "classification": task_classification.to_dict()
                })

            if task_classification.is_ambiguous:
                emit({
                    "type": "completed_with_warning",
                    "label": "Ambiguous request flagged",
                    "is_warning": True,
                    "elapsed_time": time.monotonic() - start_time
                })
                return finish({
                    "status": "completed_with_warning",
                    "error": "Ambiguous request.",
                    "final_answer": "Your request appears ambiguous or underspecified. Could you please clarify your objective with more specific details?",
                    "steps": 0,
                    "classification": task_classification.to_dict()
                })

            # Clear short-term memory for new run and record user goal
            self.short_term_memory.clear()
            self.short_term_memory.add("user", goal)

            observations_list = []
            verified_tool_results = []

            # Structured Planning Phase governed by intelligent task classification and should_plan
            needs_planning = task_classification.needs_plan
            if not needs_planning:
                try:
                    needs_planning = should_plan(goal, has_image=bool(image_bytes), has_document=bool(doc_context))
                except TypeError:
                    needs_planning = should_plan(goal, has_image=bool(image_bytes))

            if needs_planning:
                _check_cancellation()
                _check_deadline()
                emit({
                    "type": "plan_start",
                    "label": "Understanding task",
                    "elapsed_time": time.monotonic() - start_time
                })
                try:
                    task_plan = generate_plan(
                        self,
                        user_goal=goal,
                        memory_context=memory_context,
                        rag_context=doc_context,
                        experience_context=experience_context,
                        has_image=bool(image_bytes),
                        has_document=bool(doc_context)
                    )
                except TypeError:
                    task_plan = generate_plan(
                        self,
                        user_goal=goal,
                        memory_context=memory_context,
                        experience_context=experience_context,
                        has_image=bool(image_bytes)
                    )

                if task_plan.needs_plan and task_plan.steps:
                    emit({
                        "type": "plan_created",
                        "label": f"Plan created ({len(task_plan.steps)} steps)",
                        "plan": task_plan.to_dict(),
                        "elapsed_time": time.monotonic() - start_time
                    })

                    # Dependency-Aware Adaptive Step Execution Loop
                    while not task_plan.is_complete() and step < self.max_steps:
                        _check_cancellation()
                        _check_deadline()
                        ready_steps = task_plan.get_ready_steps()
                        if not ready_steps:
                            # Block remaining pending steps if dependencies failed
                            for s in task_plan.steps:
                                if s.status == "pending":
                                    s.status = "blocked"
                            break

                        for step_item in ready_steps:
                            if step >= self.max_steps:
                                break
                            _check_cancellation()
                            _check_deadline()
                            step += 1
                            step_item.status = "running"

                            emit({
                                "type": "subtask_started",
                                "label": f"Subtask {step_item.id} started: {step_item.description}",
                                "tool": step_item.tool,
                                "elapsed_time": time.monotonic() - start_time
                            })
                            emit({
                                "type": "tool_selected",
                                "label": f"Tool selected: {step_item.tool}",
                                "tool": step_item.tool,
                                "elapsed_time": time.monotonic() - start_time
                            })
                            emit({
                                "type": "tool_start",
                                "tool": step_item.tool,
                                "label": step_item.description,
                                "state": planner.AgentState.EXECUTING,
                                "elapsed_time": time.monotonic() - start_time
                            })

                            # Construct tool input: prefer step_item.tool_input if populated, otherwise description / goal
                            if getattr(step_item, "tool_input", None) and str(step_item.tool_input).strip():
                                tool_input = str(step_item.tool_input).strip()
                            elif step_item.tool == "calculator":
                                calc_match = re.search(r"[\d\.\s\+\-\*\/\^\(\)]+", step_item.description)
                                if calc_match and len(calc_match.group(0).strip()) >= 3:
                                    tool_input = calc_match.group(0).strip()
                                else:
                                    tool_input = step_item.description
                            else:
                                tool_input = f"{step_item.description} ({goal})" if len(step_item.description) > 5 else goal
                            plan_tool_key = normalize_tool_action_key(step_item.tool, tool_input)

                            # Duplicate Execution Prevention
                            if plan_tool_key in executed_tool_actions:
                                logger.info(f"Reusing verified cached result for step {step_item.id} tool '{step_item.tool}'")
                                tool_res = executed_tool_actions[plan_tool_key]
                            else:
                                tool_res = run_tool(step_item.tool, tool_input)
                                if not is_tool_error(tool_res):
                                    executed_tool_actions[plan_tool_key] = tool_res

                            is_tool_err = is_tool_error(tool_res)

                            emit({
                                "type": "tool_executed",
                                "label": f"Tool {step_item.tool} executed ({'success' if not is_tool_err else 'error'})",
                                "tool": step_item.tool,
                                "success": not is_tool_err,
                                "elapsed_time": time.monotonic() - start_time
                            })

                            if step_item.tool == "knowledge_search":
                                for m in re.finditer(r"\[(\d+)\]\s+Source:\s+([^\n]+)", tool_res):
                                    run_sources[int(m.group(1))] = {"index": int(m.group(1)), "source_label": m.group(2).strip()}

                            # Create structured observation & evaluate outcome
                            obs = planner.create_observation(step_item.id, step_item.tool, tool_res)
                            emit({
                                "type": "step_observed",
                                "tool": step_item.tool,
                                "label": f"Observed {step_item.tool} result",
                                "state": planner.AgentState.OBSERVING,
                                "elapsed_time": time.monotonic() - start_time
                            })

                            emit({
                                "type": "result_validated",
                                "label": f"Result validated: {step_item.verification_criteria}",
                                "tool": step_item.tool,
                                "success": obs.success,
                                "elapsed_time": time.monotonic() - start_time
                            })

                            eval_res = planner.evaluate_observation(step_item, obs, task_plan, goal)
                            decision = eval_res.get("decision", "continue")

                            emit({
                                "type": "step_evaluated",
                                "tool": step_item.tool,
                                "label": f"Evaluated outcome: {decision}",
                                "state": planner.AgentState.EVALUATING,
                                "elapsed_time": time.monotonic() - start_time
                            })

                            if obs.success:
                                step_item.status = "completed"
                                step_item.result = obs.result_summary
                                verified_tool_results.append({
                                    "tool": step_item.tool,
                                    "input": tool_input,
                                    "result": step_item.result
                                })
                                observations_list.append(f"Step {step_item.id} ({step_item.description}): {tool_res}")
                                emit({
                                    "type": "subtask_completed",
                                    "label": f"Subtask {step_item.id} completed",
                                    "tool": step_item.tool,
                                    "success": True,
                                    "elapsed_time": time.monotonic() - start_time
                                })
                                emit({
                                    "type": "tool_success",
                                    "tool": step_item.tool,
                                    "label": f"{step_item.description} complete",
                                    "success": True,
                                    "elapsed_time": time.monotonic() - start_time
                                })
                            else:
                                if decision == "retry":
                                    step_item.retries += 1
                                    step_item.status = "pending"
                                    logger.info(f"Retrying step {step_item.id} (Attempt {step_item.retries}/{planner.MAX_RETRIES_PER_STEP})")
                                    emit({
                                        "type": "recovery_attempted",
                                        "label": f"Recovery attempted: retry for subtask {step_item.id}",
                                        "tool": step_item.tool,
                                        "is_warning": True,
                                        "elapsed_time": time.monotonic() - start_time
                                    })
                                    emit({
                                        "type": "step_retrying",
                                        "tool": step_item.tool,
                                        "label": f"Retrying {step_item.description}",
                                        "state": planner.AgentState.RETRYING,
                                        "elapsed_time": time.monotonic() - start_time
                                    })
                                elif decision == "replan":
                                    emit({
                                        "type": "recovery_attempted",
                                        "label": f"Recovery attempted: adapting task plan for subtask {step_item.id}",
                                        "tool": step_item.tool,
                                        "is_warning": True,
                                        "elapsed_time": time.monotonic() - start_time
                                    })
                                    emit({
                                        "type": "replanning",
                                        "label": "Adapting task plan",
                                        "state": planner.AgentState.REPLANNING,
                                        "elapsed_time": time.monotonic() - start_time
                                    })
                                    task_plan = planner.replan_task(self, task_plan, obs, memory_context=memory_context, experience_context=experience_context)
                                    break
                                else:
                                    step_item.status = "failed"
                                    observations_list.append(f"Step {step_item.id} ({step_item.description}) Failed: {tool_res}")
                                    emit({
                                        "type": "tool_error",
                                        "tool": step_item.tool,
                                        "label": f"{step_item.description} failed",
                                        "is_error": True,
                                        "elapsed_time": time.monotonic() - start_time
                                    })

                    # Verification Phase
                    _check_cancellation()
                    _check_deadline()
                    emit({
                        "type": "verification_start",
                        "label": "Verifying result",
                        "elapsed_time": time.monotonic() - start_time
                    })
                    v_res = verify_plan_execution(task_plan, verified_tool_results, user_goal=goal, doc_context=doc_context)
                    task_plan.verification = v_res
                    task_plan.status = "completed" if v_res["status"] == "passed" else "completed_with_warning"

                    emit({
                        "type": "verification_completed",
                        "label": "Verified" if v_res["status"] == "passed" else "Verification warning",
                        "success": v_res["status"] == "passed",
                        "is_warning": v_res["status"] != "passed",
                        "elapsed_time": time.monotonic() - start_time
                    })

                    # Generate final response for plan execution
                    final_prompt = (
                        f"User goal: {goal}\n\n"
                        f"Execution summary:\n" + "\n".join(observations_list) + "\n\n"
                        "Provide a comprehensive, accurate final response matching required JSON schema."
                    )
                    emit({
                        "type": "provider_start",
                        "label": f"Requesting synthesis from {self.model}",
                        "elapsed_time": time.monotonic() - start_time
                    })
                    try:
                        _check_cancellation()
                        _check_deadline()
                        raw_ans = self.call_llm(final_prompt, image=pil_img)
                        emit({
                            "type": "provider_success",
                            "label": f"Synthesis received from {self.model}",
                            "success": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        decision = self._parse_and_validate_response(raw_ans)
                        final_text = decision.get("final_answer") or decision.get("thought") or "Task completed."
                    except (CancellationError, ProviderTimeoutError, PermanentProviderError):
                        emit({
                            "type": "provider_failed",
                            "label": f"Provider request failed ({self.model})",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        raise
                    except Exception as synth_err:
                        emit({
                            "type": "provider_failed",
                            "label": f"Provider request failed: {type(synth_err).__name__}",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        final_text = "Task execution completed."

                    if v_res.get("missing_information") and final_text:
                        missing_note = "\n\n**Note on Missing Information:**\n" + "\n".join([f"- {item}" for item in v_res["missing_information"]])
                        final_text += missing_note

                    if doc_context:
                        missing_items = planner.detect_missing_information(goal, doc_context)
                        if missing_items:
                            acknowledged = any(
                                phrase in final_text.lower()
                                for phrase in ["does not", "not specified", "not mentioned", "missing", "not contain", "not provided", "not listed", "unknown", "no budget"]
                            )
                            if not acknowledged:
                                final_text += "\n\n**Note on Missing Information:**\n" + "\n".join([f"- {item}" for item in missing_items])

                    citation_report = knowledge.validate_citations(final_text, run_sources)
                    if not citation_report["is_valid"]:
                        logger.warning(f"Plan Citation Validation: {citation_report['warning']}")
                        final_text += f"\n\n[Citation Warning: {citation_report['warning']}]"

                    emit({
                        "type": "completed",
                        "label": "Completed",
                        "success": True,
                        "elapsed_time": time.monotonic() - start_time
                    })

                    return finish({
                        "status": task_plan.status,
                        "final_answer": final_text,
                        "plan": task_plan.to_dict(),
                        "steps": step,
                        "citation_validation": citation_report
                    })

            # Standard ReAct execution for direct queries
            while step < self.max_steps:
                _check_cancellation()
                _check_deadline()
                step += 1
                logger.info(f"--- ReAct Iteration Step {step}/{self.max_steps} ---")

                conv_history = "\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in self.short_term_memory.get_messages()]
                )
                obs_context = "\n\n".join(observations_list) if observations_list else "No previous tool observations."

                prompt = build_agent_prompt(
                    user_goal=goal,
                    memory_context=memory_context,
                    conversation_context=conv_history,
                    observations=obs_context,
                    document_context=doc_context
                )

                emit({
                    "type": "provider_start",
                    "label": f"Requesting completion from {self.model}",
                    "elapsed_time": time.monotonic() - start_time
                })
                try:
                    raw_response = self.call_llm(prompt, image=pil_img)
                    emit({
                        "type": "provider_success",
                        "label": f"Response received from {self.model}",
                        "success": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                except CancellationError:
                    emit({
                        "type": "provider_failed",
                        "label": "Execution cancelled",
                        "is_error": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    raise
                except Exception as api_err:
                    diag = getattr(api_err, "diagnostic", None)
                    if not diag:
                        diag = parse_provider_error(api_err, model=self.model, attempt=step)

                    cat = diag.get("error_category", "unknown")
                    http_st = diag.get("http_status")
                    clean_msg = diag.get("sanitized_message", str(api_err))

                    # Provide descriptive human-friendly label for inspector activity
                    if cat == "quota_exceeded":
                        fail_label = f"Provider quota limit reached (HTTP 429): {self.model}"
                    elif cat in ("policy_restricted", "unauthorized"):
                        fail_label = f"Provider authorization restricted (HTTP {http_st or 403})"
                    elif cat == "model_not_found":
                        fail_label = f"Model unavailable (HTTP 404): {self.model}"
                    elif cat == "service_unavailable":
                        fail_label = f"Provider service unavailable (HTTP {http_st or 503})"
                    elif cat == "timeout":
                        fail_label = f"Provider request timed out ({self.model})"
                    elif cat == "bad_request":
                        fail_label = "Invalid provider request (HTTP 400)"
                    else:
                        fail_label = f"Provider request failed: {cat}"

                    emit({
                        "type": "provider_failed",
                        "label": fail_label,
                        "is_error": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    logger.error(f"Iteration Step {step} Gemini call failed ({cat}): {clean_msg}")

                    if verified_tool_results:
                        fact_lines = []
                        for v in verified_tool_results:
                            fact_lines.append(f"• {v['tool']}: {v['result']}")
                        summary_facts = "\n".join(fact_lines)

                        warn_answer = (
                            f"Verified tool results:\n{summary_facts}\n\n"
                            f"The final AI response was interrupted by a temporary service limit."
                        )
                        emit({
                            "type": "completed_with_warning",
                            "label": "Completed with limited synthesis",
                            "success": True,
                            "is_warning": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        return finish({
                            "status": "completed_with_warning",
                            "final_answer": warn_answer,
                            "error": str(api_err),
                            "steps": step,
                            "provider_diagnostic": diag
                        })

                    if cat == "quota_exceeded":
                        emit({
                            "type": "failed",
                            "label": "Quota limit reached (HTTP 429)",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        retry_note = f" (Quota resets in: {diag['retry_after']})" if diag.get("retry_after") else ""
                        ans = (
                            f"**Google Gemini Quota Limit Reached (HTTP 429)**:\n"
                            f"The Free Tier request limit for model `{self.model}` has been reached{retry_note}.\n\n"
                            f"**Recommended Recovery Actions**:\n"
                            f"1. Open **Settings** in the left sidebar and switch to another model.\n"
                            f"2. Enable pay-as-you-go billing in Google AI Studio to unlock 1,000+ RPM.\n"
                            f"3. Or update your `GEMINI_API_KEY` in `.env`."
                        )
                        return finish({
                            "status": "degraded",
                            "error": f"Quota limit reached (HTTP 429): {clean_msg}",
                            "final_answer": ans,
                            "steps": step,
                            "provider_diagnostic": diag
                        })

                    if isinstance(api_err, ProviderTimeoutError):
                        emit({
                            "type": "timeout",
                            "label": "Execution timed out",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        return finish({
                            "status": "timeout",
                            "error": str(api_err),
                            "final_answer": "Execution timed out. Please try again or simplify the request.",
                            "steps": step,
                            "provider_diagnostic": diag
                        })

                    if cat in ("policy_restricted", "unauthorized"):
                        emit({
                            "type": "failed",
                            "label": f"Provider authorization restricted (HTTP {http_st or 403})",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        ans = (
                            f"**Google Gemini Authorization Error / Policy Restriction (HTTP {http_st or 403})**:\n"
                            f"{clean_msg}\n\n"
                            f"**Recommended Recovery Actions**:\n"
                            f"1. **Google Cloud Console**: Ensure the **Generative Language API** is enabled and allowed under API restrictions for your API key.\n"
                            f"2. **Network Egress**: Verify outbound HTTPS access to `generativelanguage.googleapis.com` is permitted.\n"
                            f"3. **Settings**: Select another model or update credentials in Settings."
                        )
                        return finish({
                            "status": "failed",
                            "error": f"Authorization restricted (HTTP {http_st or 403}): {clean_msg}",
                            "final_answer": ans,
                            "steps": step,
                            "provider_diagnostic": diag
                        })

                    if cat == "model_not_found":
                        emit({
                            "type": "failed",
                            "label": f"Model unavailable (HTTP 404): {self.model}",
                            "is_error": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                        ans = (
                            f"**Gemini Model Unavailable (HTTP 404)**:\n"
                            f"The configured model `{self.model}` is no longer available or was not found.\n\n"
                            f"**Recommended Recovery Action**:\n"
                            f"Open **Settings** and choose an active model (such as `gemini-3.5-flash-lite`)."
                        )
                        return finish({
                            "status": "failed",
                            "error": f"Model not found (HTTP 404): {clean_msg}",
                            "final_answer": ans,
                            "steps": step,
                            "provider_diagnostic": diag
                        })

                    emit({
                        "type": "failed",
                        "label": "Error encountered",
                        "is_error": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    return finish({
                        "status": "failed",
                        "error": f"Gemini API error ({cat}): {clean_msg}",
                        "final_answer": "Sorry, AURA couldn't complete that request right now.",
                        "steps": step,
                        "provider_diagnostic": diag
                    })

                try:
                    decision = self._parse_and_validate_response(raw_response)
                except ValueError as val_err:
                    logger.warning(f"Iteration Step {step} Invalid model output: {val_err}")
                    emit({
                        "type": "tool_validation_failed",
                        "label": f"Invalid action rejected: {val_err}",
                        "is_warning": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    obs = f"Step {step} Error: {val_err}. Please output valid JSON matching the required schema."
                    observations_list.append(obs)
                    self.short_term_memory.add("system", obs)
                    continue

                thought = decision.get("thought", "")
                safe_thought = redact_sensitive_text(thought)
                logger.info(f"Step {step} Decision Thought Rationale: {safe_thought}")

                if decision["type"] == "final_answer":
                    final_text = decision["final_answer"]
                    logger.info(f"Step {step} Agent achieved Final Answer.")
                    self.short_term_memory.add("assistant", f"Final Answer: {final_text}")

                    # Extract and persist durable candidate memories ONLY with explicit user consent
                    if has_explicit_memory_consent(goal):
                        try:
                            candidates = extract_candidate_memories(goal)
                            persist_extracted_memories(candidates, db_path=target_db)
                        except Exception as mem_err:
                            logger.warning(f"Explicit memory extraction error: {mem_err}")

                    if doc_context:
                        missing_items = planner.detect_missing_information(goal, doc_context)
                        if missing_items:
                            acknowledged = any(
                                phrase in final_text.lower()
                                for phrase in ["does not", "not specified", "not mentioned", "missing", "not contain", "not provided", "not listed", "unknown", "no budget"]
                            )
                            if not acknowledged:
                                final_text += "\n\n**Note on Missing Information:**\n" + "\n".join([f"- {item}" for item in missing_items])

                    citation_report = knowledge.validate_citations(final_text, run_sources)
                    if not citation_report["is_valid"]:
                        logger.warning(f"ReAct Citation Validation: {citation_report['warning']}")
                        final_text += f"\n\n[Citation Warning: {citation_report['warning']}]"

                    emit({
                        "type": "final_start",
                        "label": "Preparing response",
                        "elapsed_time": time.monotonic() - start_time
                    })
                    emit({
                        "type": "completed",
                        "label": "Completed",
                        "success": True,
                        "elapsed_time": time.monotonic() - start_time
                    })
                    return finish({
                        "status": "completed",
                        "final_answer": final_text,
                        "thought": thought,
                        "steps": step,
                        "citation_validation": citation_report
                    })

                elif decision["type"] == "action":
                    action_name = decision["action"]
                    action_input = decision["action_input"]
                    tool_label = get_tool_label(action_name)

                    input_str = str(action_input)
                    tool_action_key = normalize_tool_action_key(action_name, action_input)
                    safe_input_preview = redact_sensitive_text(input_str)[:60]
                    logger.info(
                        f"Step {step} Selected Tool: '{action_name}' | "
                        f"Input Chars: {len(input_str)} | Preview: '{safe_input_preview}'"
                    )

                    emit({
                        "type": "tool_start",
                        "tool": action_name,
                        "label": tool_label,
                        "elapsed_time": time.monotonic() - start_time
                    })

                    # Duplicate Execution Prevention: Reuse cached verified output if already executed in this run
                    if tool_action_key in executed_tool_actions:
                        logger.info(f"Step {step} Reusing verified cached result for tool '{action_name}' to avoid duplicate execution.")
                        tool_result = executed_tool_actions[tool_action_key]
                        emit({
                            "type": "tool_success",
                            "tool": action_name,
                            "label": f"{tool_label} complete (cached)",
                            "success": True,
                            "elapsed_time": time.monotonic() - start_time
                        })
                    else:
                        _check_cancellation()
                        _check_deadline()
                        tool_result = run_tool(action_name, action_input)
                        if action_name == "knowledge_search":
                            for m in re.finditer(r"\[(\d+)\]\s+Source:\s+([^\n]+)", tool_result):
                                run_sources[int(m.group(1))] = {"index": int(m.group(1)), "source_label": m.group(2).strip()}

                        is_tool_err = is_tool_error(tool_result)

                        safe_result_preview = redact_sensitive_text(tool_result)[:80]
                        logger.info(
                            f"Step {step} Tool '{action_name}' Result: "
                            f"status={'error' if is_tool_err else 'success'} | "
                            f"Chars: {len(tool_result)} | Preview: '{safe_result_preview}'"
                        )

                        if not is_tool_err:
                            executed_tool_actions[tool_action_key] = tool_result
                            verified_tool_results.append({
                                "tool": action_name,
                                "input": action_input,
                                "result": tool_result.split("\n")[0]
                            })
                            emit({
                                "type": "tool_success",
                                "tool": action_name,
                                "label": f"{tool_label} complete",
                                "success": True,
                                "elapsed_time": time.monotonic() - start_time
                            })
                        else:
                            emit({
                                "type": "tool_error",
                                "tool": action_name,
                                "label": f"{tool_label} failed",
                                "is_error": True,
                                "success": False,
                                "elapsed_time": time.monotonic() - start_time
                            })

                    obs_entry = (
                        f"Step {step} Observation:\n"
                        f"Tool: {action_name}\n"
                        f"Input: {action_input}\n"
                        f"Result: {tool_result}"
                    )
                    observations_list.append(obs_entry)
                    self.short_term_memory.add("tool", obs_entry)

            logger.warning(f"Agent reached maximum reasoning steps limit ({self.max_steps}).")
            emit({
                "type": "max_steps_reached",
                "label": "Reasoning limit reached",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            return finish({
                "status": "max_steps_reached",
                "final_answer": "The agent reached its maximum reasoning steps before completing the task.",
                "steps": self.max_steps
            })

        except CancellationError as ce:
            logger.warning(f"AURA Agent execution cancelled: {ce}")
            emit({
                "type": "cancelled",
                "label": "Execution cancelled",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            return finish({
                "status": "cancelled",
                "error": str(ce),
                "final_answer": "Operation was cancelled.",
                "steps": step
            })
        except ProviderTimeoutError as te:
            logger.error(f"AURA Agent execution timed out: {te}")
            diag = getattr(te, "diagnostic", None) or parse_provider_error(te, model=self.model, attempt=step)
            emit({
                "type": "timeout",
                "label": "Execution timed out",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            return finish({
                "status": "timeout",
                "error": str(te),
                "final_answer": "Execution timed out. Please try again or simplify the request.",
                "steps": step,
                "provider_diagnostic": diag
            })
        except ProviderQuotaError as qe:
            logger.error(f"AURA Agent provider quota exhausted: {qe}")
            diag = getattr(qe, "diagnostic", None) or parse_provider_error(qe, model=self.model, attempt=step)
            emit({
                "type": "failed",
                "label": "Quota limit reached (HTTP 429)",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            retry_note = f" (Quota resets in: {diag['retry_after']})" if diag.get("retry_after") else ""
            ans = (
                f"**Google Gemini Quota Limit Reached (HTTP 429)**:\n"
                f"The Free Tier request limit for model `{self.model}` has been reached{retry_note}.\n\n"
                f"**Recommended Recovery Actions**:\n"
                f"1. Open **Settings** in the left sidebar and switch to another model.\n"
                f"2. Enable pay-as-you-go billing in Google AI Studio to unlock 1,000+ RPM.\n"
                f"3. Or update your `GEMINI_API_KEY` in `.env`."
            )
            return finish({
                "status": "degraded",
                "error": str(qe),
                "final_answer": ans,
                "steps": step,
                "provider_diagnostic": diag
            })
        except PermanentProviderError as pe:
            logger.error(f"AURA Agent permanent provider failure: {pe}")
            diag = getattr(pe, "diagnostic", None) or parse_provider_error(pe, model=self.model, attempt=step)
            err_str = str(pe)
            is_policy_403 = diag.get("error_category") == "policy_restricted" or any(k in err_str.lower() for k in ["403", "not allowed by policy", "forbidden", "permission_denied"])

            if is_policy_403:
                lbl = "Provider authorization restricted (HTTP 403)"
                clean_ans = (
                    "Provider authorization failed (HTTP 403 Forbidden): The request to Google Gemini was rejected by policy on generativelanguage.googleapis.com. "
                    "This typically occurs when API key restrictions in Google Cloud Console do not permit the Generative Language API, "
                    "or when the environment's network egress policy blocks external AI endpoints."
                )
            else:
                lbl = "Provider authorization error"
                clean_ans = f"Service configuration or authorization error: {pe}"

            emit({
                "type": "provider_error",
                "label": lbl,
                "is_error": True,
                "elapsed_time": round(time.monotonic() - start_time, 2)
            })
            return finish({
                "status": "failed",
                "error": str(pe),
                "final_answer": clean_ans,
                "steps": step,
                "error_type": "authorization_error" if is_policy_403 else "provider_error",
                "provider_diagnostic": diag
            })
        except Exception as e:
            logger.error(f"AURA Agent unexpected execution failure: {e}", exc_info=True)
            emit({
                "type": "failed",
                "label": "Execution error",
                "is_error": True,
                "elapsed_time": time.monotonic() - start_time
            })
            return finish({
                "status": "failed",
                "error": str(e),
                "final_answer": f"Agent encountered an unexpected error: {e}",
                "steps": step
            })


    def call_llm(self, prompt: str, image: Optional[Any] = None) -> str:
        """
        Hook method for model calls, allowing straightforward mocking in unit tests.
        """
        if os.environ.get("AURA_TEST_MOCK_LLM") == "1":
            p_low = prompt.lower()
            goal_part = prompt.split("=== CURRENT USER GOAL ===")[-1].lower() if "=== CURRENT USER GOAL ===" in prompt else p_low
            if "officer" in goal_part or "compliance" in goal_part or "vance" in goal_part:
                return json.dumps({
                    "thought": "Extract compliance officer from attached document context.",
                    "final_answer": "The assigned primary officer for Project Alpha compliance is Dr. Evelyn Vance. [1] Policy_Alpha.docx"
                })
            elif "retention" in goal_part or "period" in goal_part or "years" in goal_part:
                return json.dumps({
                    "thought": "Extract retention period from attached document context.",
                    "final_answer": "The retention period for Project Alpha data is exactly 14 years before decommissioning. [1] Policy_Alpha.docx"
                })
            return json.dumps({
                "thought": "Task completed.",
                "final_answer": "Document content verified successfully. [1] Policy_Alpha.docx"
            })
        return self._call_gemini_with_retry(prompt, image=image)

