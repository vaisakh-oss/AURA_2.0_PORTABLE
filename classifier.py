"""
AURA Intelligent Request Classification Subsystem
Distinguishes request categories with high fidelity to prevent unnecessary tool invocation,
route multi-step tasks to the structured planning engine, and enforce safety boundaries.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskCategory(str, Enum):
    SIMPLE_QUESTION = "simple_question"
    CALCULATION = "calculation"
    RESEARCH = "research"
    DOCUMENT_ANALYSIS = "document_analysis"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    MULTI_STEP = "multi_step"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class TaskClassification:
    """Structured representation of task classification results."""

    def __init__(
        self,
        category: TaskCategory,
        needs_plan: bool,
        suggested_tools: List[str],
        confidence: float = 1.0,
        is_ambiguous: bool = False,
        is_unsupported: bool = False,
        reasoning: str = "",
        subtask_breakdown: Optional[List[str]] = None
    ):
        self.category = category
        self.needs_plan = bool(needs_plan)
        self.suggested_tools = list(suggested_tools)
        self.confidence = float(confidence)
        self.is_ambiguous = bool(is_ambiguous)
        self.is_unsupported = bool(is_unsupported)
        self.reasoning = str(reasoning)
        self.subtask_breakdown = list(subtask_breakdown or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "needs_plan": self.needs_plan,
            "suggested_tools": self.suggested_tools,
            "confidence": self.confidence,
            "is_ambiguous": self.is_ambiguous,
            "is_unsupported": self.is_unsupported,
            "reasoning": self.reasoning,
            "subtask_breakdown": self.subtask_breakdown
        }


# Adversarial and unsupported patterns (destructive commands & exploits)
UNSUPPORTED_PATTERNS = [
    re.compile(r"\b(rm\s+-rf|drop\s+database|nuke\s+db|delete\s+all\s+files)\b", re.IGNORECASE),
    re.compile(r"^\s*(drop\s+table|alter\s+table|truncate\s+table|drop\s+database|format\s+[a-z]:|mkfs|dd\s+if=|:\(\)\{ :\|:& \};:)", re.IGNORECASE),
    re.compile(r"\b(execute|run|invoke)\s+(drop\s+table|truncate\s+table|alter\s+table|drop\s+database|rm\s+-rf|mkfs)\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\s+[a-zA-Z0-9_]+", re.IGNORECASE),
    re.compile(r"\b(format\s+c:|dd\s+if=|:\(\)\{ :\|:& \};:)\b", re.IGNORECASE),
    re.compile(r"\b(hack\s+into|exploit\s+vulnerability|sql\s+injection\s+attack|bypass\s+root)\b", re.IGNORECASE),
    re.compile(r"\b(send\s+email\s+to|initiate\s+wire\s+transfer|make\s+phone\s+call)\b", re.IGNORECASE),
]

# Patterns indicating an informational request/search about security rather than command execution
INFORMATIONAL_META_PATTERNS = [
    re.compile(r"\b(search\s+knowledge\s+base|search\s+my\s+documents|search\s+for|what\s+is|what\s+are|explain|describe)\b", re.IGNORECASE),
    re.compile(r"\b(list\s+whether|is\s+.*(?:prohibited|allowed|supported)|are\s+.*(?:prohibited|allowed|supported))\b", re.IGNORECASE),
    re.compile(r"\b(check\s+if|check\s+whether|tell\s+me\s+about|why\s+is)\b", re.IGNORECASE),
]

# Ambiguous patterns
AMBIGUOUS_PATTERNS = [
    re.compile(r"^(do\s+it|fix\s+it|fix|help|what|why|tell\s+me|more|continue|run|start|execute|try|test|go|again|how|check)[\.\?!]?$", re.IGNORECASE),
    re.compile(r"^(yes|no|maybe|ok|okay|please|sure)[\.\?!]?$", re.IGNORECASE),
]

# Math patterns
CALC_PATTERNS = [
    re.compile(r"^what\s+is\s+[\d\.\s\+\-\*\/\^\(\)\%sqrtpie]+\??$", re.IGNORECASE),
    re.compile(r"^calculate\s+[\d\.\s\+\-\*\/\^\(\)\%sqrtpie]+$", re.IGNORECASE),
    re.compile(r"^[\d\.\s\+\-\*\/\^\(\)\%]+$", re.IGNORECASE),
    re.compile(r"^(compute|evaluate)\s+[\d\.\s\+\-\*\/\^\(\)\%]+$", re.IGNORECASE),
    re.compile(r"what\s+is\s+(\d+(?:\.\d+)?%?\s*(?:of|\+|\-|\*|\/)\s*\d+(?:\.\d+)?)", re.IGNORECASE),
]

# Multi-step conjunctions
MULTI_STEP_CONJUNCTIONS = [
    re.compile(r"\b(and\s+then|then\s+calculate|after\s+that|and\s+save|search\s+and|calculate\s+and|find\s+and\s+calculate)\b", re.IGNORECASE),
    re.compile(r"\b(then\s+(?:round|divide|multiply|add|subtract|sum|split|convert|deduct|total|determine|find|compute|calculate|apply|show|combine|aggregate|express))\b", re.IGNORECASE),
    re.compile(r"\b(first\s+.*,\s*(?:then|next|secondly))\b", re.IGNORECASE),
    re.compile(r"\b(step\s*1\b.*step\s*2\b)", re.IGNORECASE),
    re.compile(r"\b(identify\s+.*extract\s+.*calculate\s+.*summarize)\b", re.IGNORECASE),
    re.compile(r"\b(show\s+intermediate\b|step\s+by\s+step\b)", re.IGNORECASE),
]


def classify_task(
    user_goal: str,
    has_image: bool = False,
    has_document: bool = False,
    has_notes: bool = False
) -> TaskClassification:
    """
    Classify a user request into an explicit TaskCategory with suggested tools
    and planning requirement indicator.

    Guarantees:
    - Never invokes tools for simple conversational or general knowledge questions.
    - Prevents executing dangerous or unsupported operations.
    - Flags ambiguous requests that require clarification or bounded handling.
    - Detects multi-step workflows requiring structured planning.
    - Accurately identifies document questions vs external research vs math.
    """
    if not user_goal or not str(user_goal).strip():
        return TaskClassification(
            category=TaskCategory.AMBIGUOUS,
            needs_plan=False,
            suggested_tools=[],
            is_ambiguous=True,
            reasoning="Empty request string provided."
        )

    clean_goal = str(user_goal).strip()
    lower_goal = clean_goal.lower()

    # 1. Check for Unsupported / Unsafe Operations
    is_info_request = any(pat.search(clean_goal) for pat in INFORMATIONAL_META_PATTERNS)
    for pat in UNSUPPORTED_PATTERNS:
        if pat.search(clean_goal):
            # If the user is asking an informational question/search about security policies or prohibited commands, do not block
            if is_info_request and not re.search(r"^\s*(drop\s+table|alter\s+table|truncate\s+table|rm\s+-rf|mkfs|format\s+[a-z]:)", clean_goal, re.IGNORECASE):
                continue
            return TaskClassification(
                category=TaskCategory.UNSUPPORTED,
                needs_plan=False,
                suggested_tools=[],
                is_unsupported=True,
                confidence=0.99,
                reasoning="Request contains unsupported system, destructive, or unauthorized actions."
            )

    # 2. Check for Ambiguous Requests
    words = clean_goal.split()
    if len(words) <= 2:
        for pat in AMBIGUOUS_PATTERNS:
            if pat.search(clean_goal):
                return TaskClassification(
                    category=TaskCategory.AMBIGUOUS,
                    needs_plan=False,
                    suggested_tools=[],
                    is_ambiguous=True,
                    confidence=0.95,
                    reasoning="Request is underspecified or lacking actionable context."
                )

    # 2b. Conversational greetings / courtesy phrases
    if re.search(r"^(hello|hi|hey|good morning|good afternoon|good evening|how are you|thank you|thanks)\b", lower_goal) and len(words) <= 12:
        return TaskClassification(
            category=TaskCategory.SIMPLE_QUESTION,
            needs_plan=False,
            suggested_tools=[],
            confidence=0.99,
            reasoning="Conversational greeting or courtesy statement answerable without tools."
        )

    # 3. Check for Multi-Step Planning Requirements
    is_multi_step = False
    subtasks = []

    for pat in MULTI_STEP_CONJUNCTIONS:
        if pat.search(clean_goal):
            is_multi_step = True
            break

    # Complex multi-clause requests with multiple operational verbs
    verbs = [
        "search", "find", "extract", "calculate", "compute", "summarize", "save", "compare",
        "check", "retrieve", "list", "identify", "determine", "count", "show", "round", "divide", "multiply", "add"
    ]
    matched_verbs = [v for v in verbs if re.search(rf"\b{v}\b", lower_goal)]
    if len(matched_verbs) >= 2 and len(words) > 8:
        is_multi_step = True

    if is_multi_step:
        suggested = []
        if any(v in matched_verbs for v in ["calculate", "compute", "round", "divide", "multiply", "add", "subtract", "sum", "split", "convert", "deduct", "total"]):
            suggested.append("calculator")
        if any(v in matched_verbs for v in ["search", "find", "check", "retrieve", "list", "identify"]):
            if has_document or "in the document" in lower_goal or "in the attached" in lower_goal:
                pass  # Document analysis handles directly
            elif "knowledge" in lower_goal or "uploaded file" in lower_goal:
                suggested.append("knowledge_search")
            else:
                suggested.append("web_search")
        if "save" in matched_verbs or "note" in lower_goal:
            suggested.append("save_note")

        return TaskClassification(
            category=TaskCategory.MULTI_STEP,
            needs_plan=True,
            suggested_tools=suggested,
            confidence=0.95,
            reasoning=f"Request contains multi-operation workflow ({', '.join(matched_verbs)}).",
            subtask_breakdown=[f"Execute operation for: {v}" for v in matched_verbs]
        )

    # 4. Attached Document Analysis
    if has_document or any(phrase in lower_goal for phrase in [
        "attached document", "this document", "uploaded report",
        "in the policy", "according to the document", "uploaded docx"
    ]):
        # Check if it requires complex multi-step document extraction
        is_complex_doc = any(phrase in lower_goal for phrase in [
            "methodology and", "extract all", "compare sections",
            "calculate timeline", "and summarize"
        ])
        return TaskClassification(
            category=TaskCategory.DOCUMENT_ANALYSIS,
            needs_plan=is_complex_doc,
            suggested_tools=[],  # Direct analysis from prompt context
            confidence=0.98,
            reasoning="Request targets attached document context."
        )

    # 5. Image / Visual Inspection
    if has_image or any(phrase in lower_goal for phrase in ["this image", "uploaded image", "attached screenshot"]):
        return TaskClassification(
            category=TaskCategory.DOCUMENT_ANALYSIS,
            needs_plan=len(words) > 10,
            suggested_tools=["vision"],
            confidence=0.95,
            reasoning="Request involves visual image analysis."
        )

    # 6. Direct Mathematical Calculation
    for pat in CALC_PATTERNS:
        if pat.search(clean_goal) or (len(words) <= 7 and pat.search(clean_goal)):
            return TaskClassification(
                category=TaskCategory.CALCULATION,
                needs_plan=False,
                suggested_tools=["calculator"],
                confidence=0.98,
                reasoning="Request is a direct mathematical expression or calculation."
            )

    # 7. Web Research
    if any(phrase in lower_goal for phrase in [
        "search web for", "google for", "find online", "latest news on",
        "current weather in", "stock price of", "search internet for",
        "search the web", "recent developments in", "advancements in",
        "weather in", "stock price"
    ]) or re.search(r"\b(search the web|search web|browse web|search online)\b", lower_goal):
        return TaskClassification(
            category=TaskCategory.RESEARCH,
            needs_plan=False,
            suggested_tools=["web_search"],
            confidence=0.90,
            reasoning="Request targets current, dynamic, or public external information."
        )

    # 8. Conversational Arithmetic with Contextual Keywords (requires digits/math operators)
    MATH_KEYWORDS = [
        "tip", "bill", "tax", "cost", "discount", "interest", "percent",
        "perimeter", "area", "volume", "radius", "diameter", "loan", "payment",
        "rate", "watts", "energy", "miles", "speed", "average", "square root"
    ]
    HAS_NUMBERS_OR_MATH = bool(re.search(r"\d", clean_goal)) and bool(re.search(r"[\d\$\%\+\-\*\/\^]", clean_goal))
    if HAS_NUMBERS_OR_MATH and any(v in lower_goal for v in ["calculate", "compute", "what is", "evaluate", "find the", "how much is", "split"]):
        if any(k in lower_goal for k in MATH_KEYWORDS) or re.search(r"\d+\s*[\+\-\*\/]\s*\d+", clean_goal):
            return TaskClassification(
                category=TaskCategory.CALCULATION,
                needs_plan=False,
                suggested_tools=["calculator"],
                confidence=0.95,
                reasoning="Conversational single-step calculation or applied arithmetic."
            )

    # 9. Knowledge Base Retrieval
    if any(phrase in lower_goal for phrase in [
        "knowledge base", "internal docs", "company policy in knowledge",
        "search my documents", "in uploaded knowledge", "knowledge store"
    ]):
        return TaskClassification(
            category=TaskCategory.KNOWLEDGE_RETRIEVAL,
            needs_plan=False,
            suggested_tools=["knowledge_search"],
            confidence=0.95,
            reasoning="Explicitly references knowledge base search."
        )

    # 8. Web Research
    if any(phrase in lower_goal for phrase in [
        "search web for", "google for", "find online", "latest news on",
        "current weather in", "stock price of", "search internet for",
        "search the web", "recent developments in", "advancements in"
    ]) or re.search(r"\b(search the web|search web|browse web|search online)\b", lower_goal):
        return TaskClassification(
            category=TaskCategory.RESEARCH,
            needs_plan=False,
            suggested_tools=["web_search"],
            confidence=0.90,
            reasoning="Request targets current, dynamic, or public external information."
        )

    # 9. Simple Direct Question (Default)
    # Greetings, general definitions, historical facts, explanations
    return TaskClassification(
        category=TaskCategory.SIMPLE_QUESTION,
        needs_plan=False,
        suggested_tools=[],
        confidence=0.90,
        reasoning="General question or statement answerable directly from model knowledge without tools."
    )
