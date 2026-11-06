import ast
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from ddgs import DDGS
from config import (
    NOTES_FILE,
    ENABLE_PYTHON_EXEC,
    ENABLE_DOCKER_SANDBOX,
    MAX_CALCULATOR_LENGTH,
    MAX_CALCULATOR_EXPONENT,
    MAX_CALCULATOR_DEPTH,
    TOOL_TIMEOUT_SECONDS,
)


# =====================================================================
# 1. WEB SEARCH TOOL
# =====================================================================
def web_search(query: str) -> str:
    """
    Search DuckDuckGo using the ddgs library with bounded network timeout.
    Returns up to 5 compact results (title, summary, URL).
    """
    if not query or not query.strip():
        return "SEARCH ERROR: Search query cannot be empty."

    query_str = query.strip()

    try:
        with DDGS(timeout=int(TOOL_TIMEOUT_SECONDS)) as ddgs:
            results = list(ddgs.text(query_str, max_results=5))

        if not results:
            return f"No search results found for query: '{query_str}'."

        formatted_results = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            url = item.get("href", item.get("link", ""))
            snippet = item.get("body", item.get("snippet", ""))
            formatted_results.append(
                f"{idx}. Title: {title}\n   URL: {url}\n   Snippet: {snippet}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"SEARCH ERROR: Search failed: {e}"


# =====================================================================
# 2. SAFE CALCULATOR TOOL (AST-BASED WITH RESOURCE LIMITS)
# =====================================================================
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": getattr(math, "tau", 2 * math.pi),
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "fabs": math.fabs,
}


def _check_ast_depth(node, depth: int = 1, max_depth: int = 20) -> None:
    """Recursively verify AST nesting depth to prevent stack overflow or recursion DoS."""
    if depth > max_depth:
        raise ValueError(f"Expression is too deeply nested (maximum depth: {max_depth}).")
    for child in ast.iter_child_nodes(node):
        _check_ast_depth(child, depth + 1, max_depth)


def _eval_ast_node(node):
    """Recursively evaluates allowed AST nodes safely with strict resource bounds."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            if abs(node.value) > 1e300:
                raise ValueError("Constant value exceeds maximum safe limit (1e300).")
            return node.value
        raise ValueError(f"Constant value '{node.value}' is not a valid number.")

    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown or unauthorized identifier '{node.id}'.")

    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            res = +operand
        elif isinstance(node.op, ast.USub):
            res = -operand
        else:
            raise ValueError(f"Unary operator '{type(node.op).__name__}' not allowed.")
        if isinstance(res, (int, float)) and abs(res) > 1e300:
            raise ValueError("Numeric value exceeds maximum safe limit (1e300).")
        return res

    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        if isinstance(node.op, ast.Add):
            res = left + right
        elif isinstance(node.op, ast.Sub):
            res = left - right
        elif isinstance(node.op, ast.Mult):
            if abs(left) > 1e150 and abs(right) > 1e150:
                raise ValueError("Multiplication magnitude exceeds safe limits (1e300).")
            res = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero.")
            res = left / right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("Division by zero.")
            res = left // right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Division by zero.")
            res = left % right
        elif isinstance(node.op, ast.Pow):
            # Enforce defensive limits on exponentiation to prevent CPU hang / DoS
            if isinstance(right, (int, float)):
                if abs(right) > MAX_CALCULATOR_EXPONENT:
                    raise ValueError(f"Exponent {right} exceeds safe calculation limit (maximum: {int(MAX_CALCULATOR_EXPONENT)}).")
                if isinstance(left, (int, float)) and abs(left) > 1 and abs(right) > 0:
                    try:
                        if abs(right) * math.log10(abs(left)) > 300:
                            raise ValueError("Resulting magnitude would exceed safe numerical limit (1e300).")
                    except (ValueError, OverflowError):
                        raise ValueError("Resulting magnitude would exceed safe numerical limit.")
            try:
                res = left ** right
            except OverflowError:
                raise ValueError("Calculation overflowed safe numeric limits.")
        else:
            raise ValueError(f"Binary operator '{type(node.op).__name__}' not allowed.")

        if isinstance(res, complex):
            raise ValueError("Complex numbers are not supported.")
        if isinstance(res, (int, float)) and (math.isinf(res) or math.isnan(res) or abs(res) > 1e300):
            raise ValueError("Numeric value exceeds maximum safe limit (1e300).")
        return res

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls (e.g. sqrt(4)) are allowed.")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_eval_ast_node(arg) for arg in node.args]
        res = SAFE_FUNCTIONS[func_name](*args)
        if isinstance(res, complex):
            raise ValueError("Complex numbers are not supported.")
        if isinstance(res, (int, float)) and (math.isinf(res) or math.isnan(res) or abs(res) > 1e300):
            raise ValueError(f"Result of '{func_name}' exceeds maximum safe numerical limit.")
        return res

    else:
        raise ValueError(f"Operation not allowed: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    Safely evaluate mathematical expressions using AST traversal.
    Blocks string evaluation, system calls, unauthorized operations,
    and enforces defensible limits on length, depth, and magnitude.
    """
    if not expression or not expression.strip():
        return "CALCULATOR ERROR: Expression cannot be empty."

    expr = expression.strip()
    if len(expr) > MAX_CALCULATOR_LENGTH:
        return f"CALCULATOR ERROR: Expression length ({len(expr)}) exceeds maximum allowed limit ({MAX_CALCULATOR_LENGTH} chars)."

    try:
        parsed = ast.parse(expr, mode="eval")
        _check_ast_depth(parsed, depth=1, max_depth=MAX_CALCULATOR_DEPTH)
        result = _eval_ast_node(parsed)
        return str(result)
    except ZeroDivisionError:
        return "CALCULATOR ERROR: Division by zero."
    except ValueError as ve:
        return f"CALCULATOR ERROR: {ve}"
    except Exception as e:
        return f"CALCULATOR ERROR: Invalid math expression ({e})."


# =====================================================================
# 3. RESTRICTED PYTHON EXECUTOR TOOL (SECURE / DISABLED BY DEFAULT)
# =====================================================================
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "ctypes", "pathlib", "builtins", "importlib", "signal",
    "threading", "multiprocessing", "asyncio"
}

BLOCKED_FUNCTIONS = {
    "eval", "exec", "compile", "__import__", "open",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "hasattr"
}

BLOCKED_ATTRIBUTES = {
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__base__", "__bases__", "__class__"
}


def _validate_python_ast(code: str):
    """
    Inspect Python AST to reject dangerous imports, built-in calls, and introspection attributes.
    Retained for static code inspection analysis.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name.split('.')[0]
                if mod_name in BLOCKED_MODULES:
                    raise ValueError(f"Importing module '{alias.name}' is prohibited.")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_name = node.module.split('.')[0]
                if mod_name in BLOCKED_MODULES:
                    raise ValueError(f"Importing from module '{node.module}' is prohibited.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_FUNCTIONS:
                raise ValueError(f"Function call '{node.func.id}' is prohibited.")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_FUNCTIONS:
                raise ValueError(f"Use of built-in '{node.id}' is prohibited.")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRIBUTES:
                raise ValueError(f"Accessing attribute '{node.attr}' is prohibited.")


def _run_docker_sandbox(code: str) -> str:
    """
    Execute Python code inside an isolated Docker container with strict boundaries:
    - Dedicated unprivileged user (nobody)
    - Dropped capabilities (ALL)
    - Read-only root filesystem
    - No network access (--network none)
    - Strict resource bounds (128MB RAM, 0.5 CPU, 30 PIDs, 5s timeout)
    - No host volume mounts, no Docker socket access, no inherited credentials

    NOTE on macOS: Docker Desktop runs inside a lightweight Linux hypervisor VM.
    Kernel-level isolation operates within the Linux VM guest kernel rather than
    native macOS kernel primitives.
    """
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return "PYTHON EXEC SECURITY ERROR: Docker executable not found in PATH."

    # Check daemon responsiveness
    try:
        check = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=3
        )
        if check.returncode != 0:
            return (
                "PYTHON EXEC SECURITY ERROR: Docker daemon is not accessible or not running. "
                "Cannot establish verified container isolation."
            )
    except Exception as err:
        return f"PYTHON EXEC SECURITY ERROR: Docker verification failed: {err}"

    # Container execution with strict isolation flags
    docker_cmd = [
        docker_bin, "run", "--rm",
        "--network", "none",
        "--read-only",
        "--user", "nobody",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--memory", "128m",
        "--cpus", "0.5",
        "--pids-limit", "30",
        "-i",
        "python:3.12-slim",
        "python", "-c", code
    ]

    try:
        timeout_sec = int(TOOL_TIMEOUT_SECONDS)
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={"PATH": "/usr/bin:/bin"}  # Strip all secrets from parent environment
        )
        outputs = []
        if proc.stdout and proc.stdout.strip():
            outputs.append(f"STDOUT:\n{proc.stdout.strip()}")
        if proc.stderr and proc.stderr.strip():
            outputs.append(f"STDERR:\n{proc.stderr.strip()}")
        if not outputs:
            return "PYTHON EXEC: Code executed successfully in isolated container with no output."
        return "\n".join(outputs)
    except subprocess.TimeoutExpired:
        return f"PYTHON EXEC ERROR: Isolated container execution timed out (limit: {timeout_sec}s)."
    except Exception as e:
        return f"PYTHON EXEC ERROR: Container execution failed ({e})."


def python_exec(code: str) -> str:
    """
    Restricted Python execution interface.
    SECURITY NOTICE:
    Host-level arbitrary Python execution is strictly DISABLED by default.
    AST validation, restricted globals, or environment variable stripping on the host
    DO NOT provide a security sandbox against dynamic Python reflection or transitive imports.
    Execution is only permitted when an explicitly configured, verified Docker container
    sandbox (ENABLE_DOCKER_SANDBOX=True) with dropped privileges and no network access
    is active and verified.
    """
    if not code or not code.strip():
        return "PYTHON EXEC ERROR: Code cannot be empty."

    # If Docker sandbox is explicitly requested, attempt verified container execution
    if ENABLE_DOCKER_SANDBOX:
        return _run_docker_sandbox(code)

    # By default, Python execution on host is strictly disabled for security
    return (
        "PYTHON EXEC SECURITY ERROR: Python execution is disabled by default for security. "
        "Host-level arbitrary code execution is prohibited without an isolated, verified container sandbox."
    )


# =====================================================================
# 4. SAVE NOTE TOOL
# =====================================================================
def save_note(note: str) -> str:
    """
    Save a text note with a timestamp to notes.json.
    """
    if not note or not note.strip():
        return "NOTE ERROR: Note content cannot be empty."

    note_text = note.strip()
    notes_path = Path(NOTES_FILE)

    # Load existing notes
    notes = []
    if notes_path.exists():
        try:
            with open(notes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    notes = data
        except Exception:
            notes = []

    # Append new note
    new_entry = {
        "note": note_text,
        "timestamp": datetime.datetime.now().isoformat()
    }
    notes.append(new_entry)

    # Write back to JSON file
    try:
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        return f"SUCCESS: Note saved successfully to {notes_path.name}."
    except Exception as e:
        return f"NOTE ERROR: Failed to save note: {e}"


# =====================================================================
# 5. READ NOTES TOOL
# =====================================================================
def read_notes(query: str = "") -> str:
    """
    Read saved notes from notes.json. Filter by query if provided.
    """
    notes_path = Path(NOTES_FILE)

    if not notes_path.exists():
        return "No notes found. (notes.json does not exist yet)"

    try:
        with open(notes_path, "r", encoding="utf-8") as f:
            notes = json.load(f)
    except Exception:
        return "NOTE ERROR: Failed to read notes.json (file may be malformed)."

    if not isinstance(notes, list) or not notes:
        return "No notes found."

    # Filter notes if query is provided
    query_str = query.strip().lower() if query else ""
    if query_str:
        filtered = [
            n for n in notes
            if isinstance(n, dict) and query_str in n.get("note", "").lower()
        ]
        if not filtered:
            return f"No notes matching query: '{query}'."
        target_notes = filtered
    else:
        target_notes = notes[-5:]  # Return up to 5 most recent notes

    formatted = []
    for idx, entry in enumerate(target_notes, 1):
        text = entry.get("note", "")
        timestamp = entry.get("timestamp", "Unknown time")
        formatted.append(f"{idx}. [{timestamp}] {text}")

    return "\n".join(formatted)


# =====================================================================
# 6. KNOWLEDGE SEARCH TOOL
# =====================================================================
def knowledge_search(query: str) -> str:
    """
    Search the user's uploaded knowledge base documents for relevant context chunks.
    Combines dense semantic vectors and lexical BM25 matching via Reciprocal Rank Fusion (RRF),
    bounded context assembly, and citation provenance.
    """
    if not query or not query.strip():
        return "KNOWLEDGE ERROR: Query cannot be empty."

    try:
        import knowledge
        results = knowledge.search_knowledge(query.strip(), top_k=5)
        if not results:
            return "No relevant document context found in your knowledge base."

        rag_ctx = knowledge.assemble_rag_context(results)
        if not rag_ctx["selected_chunks"]:
            return "No relevant document context found in your knowledge base."

        formatted_chunks = []
        for c in rag_ctx["selected_chunks"]:
            idx = c["citation_index"]
            source_label = c["source_label"]
            text = c.get("text", "").strip()
            formatted_chunks.append(f"[{idx}] Source: {source_label}\nContent: {text}")

        return "\n\n".join(formatted_chunks)
    except Exception as e:
        return f"KNOWLEDGE ERROR: Search failed ({e})."


# =====================================================================
# 7. TOOL REGISTRY, AUTHORIZATION & SAFE DISPATCHER
# =====================================================================
TOOL_REGISTRY = {
    "web_search": web_search,
    "calculator": calculator,
    "python_exec": python_exec,
    "save_note": save_note,
    "read_notes": read_notes,
    "knowledge_search": knowledge_search,
}

IDEMPOTENT_TOOLS = {
    "calculator",
    "read_notes",
    "knowledge_search",
    "web_search",
    "python_exec",
}

NON_IDEMPOTENT_TOOLS = {
    "save_note",
}


class ToolDefinition:
    """Explicit formal contract and authorization specification for an AURA tool."""

    def __init__(
        self,
        identifier: str,
        name: str,
        description: str,
        handler: Any,
        permission_level: str,
        side_effect: str,
        timeout_seconds: float,
        max_output_chars: int,
        is_idempotent: bool,
        failure_behavior: str,
        schema: Dict[str, Any]
    ):
        self.identifier = identifier
        self.name = name
        self.description = description
        self.handler = handler
        self.permission_level = permission_level  # "public", "user_storage", "restricted"
        self.side_effect = side_effect  # "read_only", "state_mutation", "network_io"
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_chars = int(max_output_chars)
        self.is_idempotent = bool(is_idempotent)
        self.failure_behavior = failure_behavior  # "return_error_string", "replan"
        self.schema = dict(schema)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "description": self.description,
            "permission_level": self.permission_level,
            "side_effect": self.side_effect,
            "timeout_seconds": self.timeout_seconds,
            "max_output_chars": self.max_output_chars,
            "is_idempotent": self.is_idempotent,
            "failure_behavior": self.failure_behavior,
            "schema": self.schema
        }


TOOL_SPECIFICATIONS: Dict[str, ToolDefinition] = {
    "web_search": ToolDefinition(
        identifier="web_search",
        name="Web Search",
        description="Search DuckDuckGo for public real-time web results.",
        handler=web_search,
        permission_level="public",
        side_effect="network_io",
        timeout_seconds=float(TOOL_TIMEOUT_SECONDS),
        max_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000")),
        is_idempotent=True,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 1, "max_length": 1000}
    ),
    "calculator": ToolDefinition(
        identifier="calculator",
        name="Calculator",
        description="Safely evaluate mathematical expressions using AST with resource limits.",
        handler=calculator,
        permission_level="public",
        side_effect="read_only",
        timeout_seconds=5.0,
        max_output_chars=1000,
        is_idempotent=True,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 1, "max_length": MAX_CALCULATOR_LENGTH}
    ),
    "python_exec": ToolDefinition(
        identifier="python_exec",
        name="Python Executor",
        description="Execute restricted Python code inside an isolated container sandbox.",
        handler=python_exec,
        permission_level="restricted",
        side_effect="read_only",
        timeout_seconds=float(TOOL_TIMEOUT_SECONDS),
        max_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000")),
        is_idempotent=True,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 1, "max_length": 10000}
    ),
    "save_note": ToolDefinition(
        identifier="save_note",
        name="Save Note",
        description="Store a text note or user preference into long-term memory storage.",
        handler=save_note,
        permission_level="user_storage",
        side_effect="state_mutation",
        timeout_seconds=5.0,
        max_output_chars=1000,
        is_idempotent=False,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 1, "max_length": 5000}
    ),
    "read_notes": ToolDefinition(
        identifier="read_notes",
        name="Read Notes",
        description="Retrieve stored notes or filter by keyword query from notes storage.",
        handler=read_notes,
        permission_level="user_storage",
        side_effect="read_only",
        timeout_seconds=5.0,
        max_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000")),
        is_idempotent=True,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 0, "max_length": 500}
    ),
    "knowledge_search": ToolDefinition(
        identifier="knowledge_search",
        name="Knowledge Search",
        description="Search user's uploaded knowledge base documents via hybrid vector and lexical retrieval.",
        handler=knowledge_search,
        permission_level="user_storage",
        side_effect="read_only",
        timeout_seconds=float(TOOL_TIMEOUT_SECONDS),
        max_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000")),
        is_idempotent=True,
        failure_behavior="return_error_string",
        schema={"type": "string", "min_length": 1, "max_length": 1000}
    ),
}


def validate_tool_authorization(tool_name: str) -> Tuple[bool, Optional[str]]:
    """
    Enforce application-level tool authorization.
    Verifies that the requested tool is explicitly registered in TOOL_SPECIFICATIONS
    and that restricted tools adhere to runtime security prerequisites.
    """
    if not tool_name or not str(tool_name).strip():
        return False, "TOOL ERROR: Tool name cannot be empty."

    clean_name = str(tool_name).strip()
    if clean_name == "search_knowledge_base":
        clean_name = "knowledge_search"

    if clean_name not in TOOL_SPECIFICATIONS or clean_name not in TOOL_REGISTRY:
        return False, f"TOOL ERROR: Unknown tool '{clean_name}'. Allowed tools: {sorted(list(TOOL_REGISTRY.keys()))}"

    tool_def = TOOL_SPECIFICATIONS[clean_name]
    if tool_def.permission_level == "restricted":
        if clean_name == "python_exec" and not ENABLE_DOCKER_SANDBOX:
            return False, "TOOL AUTHORIZATION ERROR: 'python_exec' is restricted and disabled on the host without verified container isolation."

    return True, None


SENSITIVE_PATTERNS = [
    # Google API Key
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_API_KEY]"),
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
    before writing to logs, observations, or telemetry.
    """
    if text is None:
        return ""
    val = str(text)
    for pattern, replacement in SENSITIVE_PATTERNS:
        val = pattern.sub(replacement, val)
    return val


def is_tool_idempotent(tool_name: str) -> bool:
    """
    Check if a tool is idempotent (safe to retry or re-read without duplicate side effects).
    Tools that mutate persistent state (like save_note) are non-idempotent.
    """
    clean = str(tool_name).strip()
    return clean in IDEMPOTENT_TOOLS and clean not in NON_IDEMPOTENT_TOOLS


def is_tool_error(result: str) -> bool:
    """
    Detect whether a tool output string indicates an execution, validation, or timeout error.
    """
    if not result or not isinstance(result, str):
        return True
    res_clean = result.strip()
    error_prefixes = (
        "TOOL ERROR", "CALCULATOR ERROR", "PYTHON EXEC ERROR",
        "NOTE ERROR", "SEARCH ERROR", "KNOWLEDGE ERROR",
        "TIMEOUT ERROR", "VALIDATION ERROR", "TOOL AUTHORIZATION ERROR"
    )
    if any(res_clean.startswith(prefix) for prefix in error_prefixes):
        return True
    if res_clean.startswith("Error:") or res_clean.startswith("Failed:"):
        return True
    return False


def validate_tool_input(tool_name: str, action_input: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate tool name, authorization permissions, and input argument structure before execution.
    Enforces maximum input length bounds and validates types.
    """
    is_auth, auth_err = validate_tool_authorization(tool_name)
    if not is_auth:
        return False, auth_err

    clean_name = str(tool_name).strip()
    if clean_name == "search_knowledge_base":
        clean_name = "knowledge_search"

    if action_input is None:
        return True, None

    if isinstance(action_input, (str, int, float, bool, dict, list)):
        input_str = str(action_input)
        if len(input_str) > 50000:
            return False, f"TOOL ERROR: Input argument for '{clean_name}' exceeds maximum allowed length (50,000 chars)."
        return True, None

    return False, f"TOOL ERROR: Invalid input argument type '{type(action_input).__name__}' for tool '{clean_name}'."


def run_tool(tool_name: str, action_input: Any = "") -> str:
    """
    Safely execute a registered tool:
    - Validates tool authorization and input arguments before execution.
    - Enforces bounded tool outputs (MAX_TOOL_OUTPUT_CHARS).
    - Redacts sensitive secrets, API keys, and credentials from outputs and error messages.
    - Preserves security controls, sandbox restrictions, and untrusted data boundaries.
    """
    is_valid, err_msg = validate_tool_input(tool_name, action_input)
    if not is_valid:
        return redact_sensitive_text(err_msg)

    clean_name = str(tool_name).strip()
    if clean_name == "search_knowledge_base":
        clean_name = "knowledge_search"
    tool_func = TOOL_REGISTRY[clean_name]

    try:
        if isinstance(action_input, dict):
            raw_out = str(tool_func(**action_input))
        else:
            raw_out = str(tool_func(action_input))
    except Exception as e:
        raw_out = f"TOOL ERROR executing '{clean_name}': {e}"

    # Redact any accidental secrets in tool output or error message
    clean_out = redact_sensitive_text(raw_out)

    # Enforce strict output character bounds
    max_output = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000"))
    if len(clean_out) > max_output:
        notice = f"\n\n[... Tool '{clean_name}' output truncated at {max_output} characters ...]"
        clean_out = clean_out[:max_output - len(notice)] + notice

    return clean_out
