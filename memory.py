import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import MEMORY_FILE
import storage

VALID_MEMORY_TYPES = {
    "USER_PREFERENCE",
    "USER_PROFILE",
    "PROJECT_CONTEXT",
    "TASK_CONTEXT",
    "FACT"
}

SENSITIVE_PATTERNS = [
    re.compile(r"(?i)\b(api_?key|password|passwd|secret|auth_token|bearer\s+[a-z0-9_-]+|private_key|credit_card|ssn)\b"),
    re.compile(r"(?i)\b(sk-[a-zA-Z0-9]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|ghp_[a-zA-Z0-9]{36})\b"),
    re.compile(r"(?i)my\s+(api\s*key|password|secret|token)\s+is\b")
]

INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)reveal\s+(the\s+)?(gemini\s+)?api\s*key"),
    re.compile(r"(?i)system\s+instruction"),
    re.compile(r"(?i)override\s+(safety|system|rules)"),
    re.compile(r"(?i)use\s+python_exec\s+to\s+read")
]


# =====================================================================
# 1. SHORT-TERM MEMORY (IN-MEMORY CONVERSATION & EVENT BUFFER)
# =====================================================================
class ShortTermMemory:
    """
    In-memory buffer for tracking conversation turns and agent execution steps
    during a single run. Discarded when the application finishes.
    """

    def __init__(self):
        self.messages = []

    def add(self, role: str, content: str):
        """Add a new message or event turn (e.g., user, assistant, system, tool)."""
        if role and content:
            self.messages.append({
                "role": str(role).strip(),
                "content": str(content).strip()
            })

    def get_messages(self) -> list:
        """Return all recorded messages for the current run."""
        return list(self.messages)

    def clear(self):
        """Reset and clear short-term memory buffer."""
        self.messages.clear()


# =====================================================================
# 2. LONG-TERM MEMORY (PERSISTENT JSON STORAGE & COMPATIBILITY)
# =====================================================================
def load_memory(file_path=MEMORY_FILE) -> list:
    """
    Load persistent memory facts from memory.json.
    Returns an empty list if the file is missing or malformed.
    """
    target_path = Path(file_path)

    if not target_path.exists():
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, OSError):
        return []


def save_memory_fact(fact: str, file_path=MEMORY_FILE, db_path: Optional[str] = None) -> dict:
    """
    Save an explicit fact with a UTC ISO timestamp to memory.json & SQLite memories table.
    Preserves existing stored memories.
    """
    if not fact or not str(fact).strip():
        return {}

    fact_text = str(fact).strip()
    target_path = Path(file_path)

    memories = load_memory(target_path)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_entry = {
        "fact": fact_text,
        "timestamp": timestamp
    }

    memories.append(new_entry)

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2)
    except Exception as e:
        print(f"[MEMORY ERROR] Failed to save memory fact: {e}")

    # Persist into SQLite structured memories table
    # Only write to production db if using the default memory file or explicitly requested
    effective_db = db_path if db_path is not None else (storage.DB_FILE if target_path == Path(MEMORY_FILE) else None)
    if effective_db:
        try:
            storage.add_memory(
                content=fact_text,
                memory_type="FACT",
                importance=0.6,
                confidence=0.9,
                source="user_explicit",
                db_path=effective_db
            )
        except Exception:
            pass

    return new_entry


def memory_as_text(limit: int = 10, file_path=MEMORY_FILE, query: str = "", db_path: Optional[str] = None) -> str:
    """
    Format relevant long-term memories into a compact, safe string suitable
    for injecting into the LLM system prompt.
    """
    effective_db = db_path if db_path is not None else storage.get_default_db_file()
    rel = get_relevant_memories(query=query, limit=limit, db_path=effective_db)
    return format_memories_for_prompt(rel)


# =====================================================================
# 3. INTELLIGENT MEMORY EXTRACTION & RETRIEVAL (STEP 14)
# =====================================================================
def _normalize_text_tokens(text: str) -> set:
    """Helper to convert text into a normalized set of lowercase words for deduplication."""
    clean = re.sub(r"[^\w\s]", "", str(text).lower())
    stop_words = {"a", "an", "the", "is", "are", "am", "my", "your", "this", "that", "it", "to", "for", "in", "of", "and", "or", "called", "named"}
    tokens = {w for w in clean.split() if w not in stop_words and len(w) > 1}
    return tokens


def extract_candidate_memories(user_message: str, assistant_context: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Deterministically extract durable, high-quality candidate memories from a user message.
    Rejects casual chatter, sensitive secrets/API keys, and prompt injection attempts.
    """
    if not user_message or not str(user_message).strip():
        return []

    text = str(user_message).strip()

    # Security Filter 1: Reject sensitive information / credentials / secrets
    for pat in SENSITIVE_PATTERNS:
        if pat.search(text):
            return []

    # Security Filter 2: Reject adversarial prompt injection instructions
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return []

    # Filter 3: Reject casual conversational noise
    lower_text = text.lower()
    casual_greetings = {"hello", "hi", "hey", "good morning", "good evening", "how are you", "thanks", "thank you", "bye", "ok", "yes", "no"}
    if lower_text in casual_greetings or len(text.split()) < 2:
        return []

    candidates = []

    # Category 1: USER_PROFILE ("Call me Pratik", "My name is Pratik", "I am an AIML student", "My background is...")
    profile_match = re.search(r"(?i)\b(?:call me|my name is)\s+([A-Za-z0-9_-]+)", text)
    if profile_match:
        name = profile_match.group(1).capitalize()
        candidates.append({
            "content": f"User's name is {name}",
            "memory_type": "USER_PROFILE",
            "importance": 0.85,
            "confidence": 0.95,
            "source": "user_explicit"
        })

    role_match = re.search(r"(?i)\bi am (?:an?|a)\s+([A-Za-z0-9_\s]{2,30}?)(?:\.|$|,|\s+and)", text)
    if role_match and not profile_match and "prefer" not in lower_text:
        role_str = role_match.group(1).strip()
        if role_str.lower() not in {"here", "there", "ready", "sure", "happy", "asking"}:
            candidates.append({
                "content": f"User is {role_str}",
                "memory_type": "USER_PROFILE",
                "importance": 0.75,
                "confidence": 0.85,
                "source": "user_explicit"
            })

    # Category 2: USER_PREFERENCE ("I prefer concise answers", "I prefer Python", "I like dark mode")
    pref_match = re.search(r"(?i)\bi prefer\s+([A-Za-z0-9_\s]{2,40}?)(?:\.|$|,)", text)
    if pref_match:
        pref_str = pref_match.group(1).strip()
        candidates.append({
            "content": f"User prefers {pref_str}",
            "memory_type": "USER_PREFERENCE",
            "importance": 0.75,
            "confidence": 0.90,
            "source": "user_explicit"
        })
    elif "concise" in lower_text and "answer" in lower_text:
        candidates.append({
            "content": "User prefers concise answers",
            "memory_type": "USER_PREFERENCE",
            "importance": 0.75,
            "confidence": 0.90,
            "source": "user_explicit"
        })

    # Category 3: PROJECT_CONTEXT ("My project is AURA", "Current project uses Streamlit", "Building an AI assistant")
    proj_match = re.search(r"(?i)\b(?:my project is|my current project is|project is named|project is called)\s+([A-Za-z0-9_\s]{2,30}?)(?:\.|$|,)", text)
    if proj_match:
        proj_name = proj_match.group(1).strip()
        candidates.append({
            "content": f"Current project is {proj_name}",
            "memory_type": "PROJECT_CONTEXT",
            "importance": 0.85,
            "confidence": 0.95,
            "source": "user_explicit"
        })

    proj_tech_match = re.search(r"(?i)\b(?:project uses|stack is|built with)\s+([A-Za-z0-9_\s\+]{2,30}?)(?:\.|$|,)", text)
    if proj_tech_match:
        tech_str = proj_tech_match.group(1).strip()
        candidates.append({
            "content": f"Project uses {tech_str}",
            "memory_type": "PROJECT_CONTEXT",
            "importance": 0.80,
            "confidence": 0.90,
            "source": "user_explicit"
        })

    # Category 4: TASK_CONTEXT ("Current task is Step 14", "Active constraint: no external DB")
    task_match = re.search(r"(?i)\b(?:current task is|pending objective is|our goal is)\s+([A-Za-z0-9_\s]{2,40}?)(?:\.|$|,)", text)
    if task_match:
        task_str = task_match.group(1).strip()
        candidates.append({
            "content": f"Task objective: {task_str}",
            "memory_type": "TASK_CONTEXT",
            "importance": 0.65,
            "confidence": 0.80,
            "source": "user_explicit"
        })

    # Category 5: FACT ("Remember that X", "Fact: X", "Save note: X")
    fact_match = re.search(r"(?i)\b(?:remember that|remember:|save note:?|fact:)\s+([A-Za-z0-9_\s\.\,-]{3,80})", text)
    if fact_match and not candidates:
        fact_str = fact_match.group(1).strip()
        candidates.append({
            "content": fact_str,
            "memory_type": "FACT",
            "importance": 0.60,
            "confidence": 0.85,
            "source": "user_explicit"
        })

    return candidates


def persist_extracted_memories(
    candidates: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    confidence_threshold: float = 0.6,
    importance_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Persist candidate memories above threshold into SQLite, avoiding duplicate active memories.
    """
    if not candidates:
        return []

    effective_db = db_path if db_path is not None else storage.get_default_db_file()
    existing_memories = storage.list_memories(active_only=True, db_path=effective_db)
    persisted = []

    for cand in candidates:
        conf = float(cand.get("confidence", 0.0))
        imp = float(cand.get("importance", 0.0))

        if conf < confidence_threshold or imp < importance_threshold:
            continue

        cand_content = str(cand.get("content", "")).strip()
        cand_tokens = _normalize_text_tokens(cand_content)
        if not cand_tokens:
            continue

        # Check for duplication against existing active memories
        is_duplicate = False
        for ex in existing_memories:
            ex_tokens = _normalize_text_tokens(ex.get("content", ""))
            if not ex_tokens:
                continue

            # Check overlap coefficient
            intersection = cand_tokens.intersection(ex_tokens)
            smaller_len = min(len(cand_tokens), len(ex_tokens))
            overlap = len(intersection) / float(smaller_len) if smaller_len > 0 else 0.0

            if overlap >= 0.85 or cand_content.lower() == ex.get("content", "").lower():
                is_duplicate = True
                # Refresh timestamp & update confidence/importance if higher
                storage.update_memory(
                    ex["id"],
                    importance=max(ex.get("importance", 0.5), imp),
                    confidence=max(ex.get("confidence", 0.8), conf),
                    db_path=effective_db
                )
                break

        if not is_duplicate:
            m_id = storage.add_memory(
                content=cand_content,
                memory_type=cand.get("memory_type", "FACT"),
                importance=imp,
                confidence=conf,
                source=cand.get("source", "user_explicit"),
                db_path=effective_db
            )
            cand["id"] = m_id
            persisted.append(cand)

    return persisted


def get_relevant_memories(
    query: str = "",
    limit: int = 5,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve and rank the most relevant structured memories considering:
    1. Relevance to query (word/token overlap or semantic match)
    2. Importance weight
    3. Confidence weight
    4. Recency & Access frequency
    """
    effective_db = db_path if db_path is not None else storage.get_default_db_file()
    active_memories = storage.list_memories(active_only=True, db_path=effective_db)

    # Fallback to old notes if memories table is empty
    if not active_memories:
        old_notes = storage.list_notes(db_path=db_path, sync_json=False)
        for n in old_notes:
            active_memories.append({
                "id": n.get("id"),
                "content": n.get("content"),
                "memory_type": "FACT",
                "importance": 0.5,
                "confidence": 0.8,
                "created_at": n.get("created_at"),
                "access_count": 0
            })

    if not active_memories:
        return []

    q_tokens = _normalize_text_tokens(query) if query else set()

    scored = []
    for m in active_memories:
        content = m.get("content", "")
        m_tokens = _normalize_text_tokens(content)

        # 1. Similarity score
        if q_tokens and m_tokens:
            intersection = q_tokens.intersection(m_tokens)
            union = q_tokens.union(m_tokens)
            sim_score = len(intersection) / float(len(union)) if union else 0.0
        else:
            sim_score = 0.5  # Neutral baseline when no query provided

        # 2. Importance & Confidence
        imp = float(m.get("importance", 0.5))
        conf = float(m.get("confidence", 0.8))

        # 3. Access bonus
        acc_count = int(m.get("access_count", 0))
        acc_bonus = min(0.1, acc_count * 0.02)

        total_score = (sim_score * 0.45) + (imp * 0.25) + (conf * 0.20) + acc_bonus
        scored.append((total_score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [item[1] for item in scored[:limit]]

    # Update access tracking
    ids_to_touch = [item["id"] for item in top_items if isinstance(item.get("id"), int)]
    if ids_to_touch:
        try:
            storage.record_memory_access(ids_to_touch, db_path=db_path)
        except Exception:
            pass

    return top_items


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """
    Format retrieved memories inside strict <stored_memory> XML tags.
    """
    if not memories:
        return "No long-term memories retrieved."

    lines = ["<stored_memory>"]
    for m in memories:
        content = m.get("content") or m.get("fact") or m.get("note", "")
        m_type = m.get("memory_type", "FACT")
        lines.append(f"• [{m_type}] {content}")
    lines.append("</stored_memory>")
    return "\n".join(lines)
