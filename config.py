import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
MEMORY_FILE = BASE_DIR / "memory.json"
NOTES_FILE = BASE_DIR / "notes.json"

# Ensure logs directory exists automatically
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API and Agent settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

# Security & Sandbox settings
ENABLE_PYTHON_EXEC = os.getenv("ENABLE_PYTHON_EXEC", "false").lower() == "true"
ENABLE_DOCKER_SANDBOX = os.getenv("ENABLE_DOCKER_SANDBOX", "false").lower() == "true"
MAX_DOCUMENT_SIZE_BYTES = int(os.getenv("MAX_DOCUMENT_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
MAX_CALCULATOR_LENGTH = int(os.getenv("MAX_CALCULATOR_LENGTH", "500"))
MAX_CALCULATOR_EXPONENT = float(os.getenv("MAX_CALCULATOR_EXPONENT", "1000"))
MAX_CALCULATOR_DEPTH = int(os.getenv("MAX_CALCULATOR_DEPTH", "20"))

# Execution Reliability, Timeouts & Retries
PROVIDER_TIMEOUT_SECONDS = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30.0"))
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "15.0"))
AGENT_RUN_TIMEOUT_SECONDS = float(os.getenv("AGENT_RUN_TIMEOUT_SECONDS", "120.0"))
MAX_TRANSIENT_RETRIES = int(os.getenv("MAX_TRANSIENT_RETRIES", "3"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000"))


def update_config(model: Optional[str] = None, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Persist configuration updates to .env and synchronize active runtime environment.
    Guarantees settings survive server restarts without exposing or modifying API keys.
    """
    global GEMINI_MODEL, MAX_STEPS
    env_file = BASE_DIR / ".env"

    if model:
        GEMINI_MODEL = str(model).strip()
        os.environ["GEMINI_MODEL"] = GEMINI_MODEL
        if env_file.exists():
            try:
                import dotenv
                dotenv.set_key(str(env_file), "GEMINI_MODEL", GEMINI_MODEL)
            except Exception:
                pass

    if max_steps is not None:
        MAX_STEPS = int(max_steps)
        os.environ["MAX_STEPS"] = str(MAX_STEPS)
        if env_file.exists():
            try:
                import dotenv
                dotenv.set_key(str(env_file), "MAX_STEPS", str(MAX_STEPS))
            except Exception:
                pass

    return {
        "model": GEMINI_MODEL,
        "max_steps": MAX_STEPS
    }
