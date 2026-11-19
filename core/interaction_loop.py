# ==============================================================================
# Program: interaction_loop.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Hot-reloadable interaction pipeline intercepting prompts and outputs.
#          Injects dynamic context, applies prompt transformations, enforces
#          offline security rules, and post-processes generations without
#          requiring application restarts.
# ==============================================================================

# THE HACKABLE LAYER
# Modify this file to change how the application interacts with the LLM.
# Hot-reloaded before every generation -- save and click Generate, no restart needed.

import logging
import os
import re
from app.engine import config_store
from app.engine.model_loader import ModelLoader
from app.utils.compactor import compact_trace_for_ai
from core.default_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    LEGACY_DEFAULT_SYSTEM_PROMPTS,
    RECENCY_INSTRUCTION as _RECENCY_INSTRUCTION,
)

# System prompt used when the custom_greeting adapter (or any adapter) is active.
# This matches the training data, so the adapter fires correctly.
logger = logging.getLogger("karl.codex_injector")


# Per-run overrides set by the caller before generation (e.g. machine speak flags); consulted by
# _machine_speak_enabled.
ACTIVE_HYPERPARAMS: dict = {}

# Minimal prompt matching what adapters were trained with.
_ADAPTER_SYSTEM_PROMPT = "Always respond in English."
_BASE_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
# Instruction appended to the system prompt when history is compressed, so the model answers in the
# same compressed dialect.
_MACHINE_SPEAK_RULE = (
    "[COMPRESSION RULE] The message history has been serialized into token-dense "
    "'Machine Speak' to save context space.\n"
    "You must respond in kind using the exact same compressed format:\n"
    "1. Shorten all response properties according to the Key Map (e.g., use 'r' "
    "for response, 'tk' for thoughts).\n"
    "2. Express all code modifications as unified diff blocks ($diff) rather "
    "than writing full files."
)

# Maps lowercase keywords found in a user message to a reference page in data/codex_library. Several
# aliases share one page (cpp/c++, js/javascript).
_CODEX_KEYWORD_MAP = {
    "python": "Python.html",
    "c++": "C++.html",
    "cpp": "C++.html",
    "sql": "SQL.html",
    "rust": "Rust.html",
    "react": "React.html",
    "node": "Nodejs.html",
    "nodejs": "Nodejs.html",
    "node.js": "Nodejs.html",
    "go": "Go.html",
    "golang": "Go.html",
    "agile": "Agile.html",
    "xcode": "Xcode.html",
    "swift": "Swift.html",
    "fortran": "Fortran.html",
    "c#": "C#.html",
    "csharp": "C#.html",
    "c": "C.html",
    "docker": "Docker.html",
    "kubernetes": "Kubernetes.html",
    "k8s": "Kubernetes.html",
    "css": "CSS.html",
    "html": "HTML.html",
    "typescript": "TypeScript.html",
    "ts": "TypeScript.html",
    "java": "Java.html",
    "javascript": "JavaScript.html",
    "js": "JavaScript.html",
    "api": "APIs.html",
    "apis": "APIs.html",
    "uvicorn": "Uvicorn.html",
    "fastapi": "FastAPI.html"
}

def strip_html_tags(html: str) -> str:
    """Convert a codex HTML page to compact plain text for prompt injection: drop script/style, turn block tags into newlines, remove remaining tags, decode the few common entities, and collapse runs of blank lines."""
    # Remove script and style elements first
    # Remove script/style bodies entirely; their text is code, not documentation.
    text = re.sub(r'<(script|style)\b[^>]*>([\s\S]*?)</\1>', '', html)
    # Replace block-level element tags with newlines or spaces to preserve spacing
    text = re.sub(r'</?(div|p|h[1-6]|li|ul|ol|table|tr|td|thead|tbody|br|pre|code)\b[^>]*>', '\n', text)
    # Remove any other tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities (optional, but good for common ones like &lt;, &gt;, &amp;, &quot;)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # Clean up excess whitespace/newlines
    lines = [line.strip() for line in text.split('\n')]
    cleaned = []
    last_empty = False
    for line in lines:
        if line:
            cleaned.append(line)
            last_empty = False
        elif not last_empty:
            cleaned.append("")
            last_empty = True
    return '\n'.join(cleaned).strip()

def matches_keyword(text: str, keyword: str) -> bool:
    """Case-insensitive keyword test with word boundaries. 'c' is special-cased so it does not fire on 'c++' or 'c#'; keywords that start or end with punctuation (c++, node.js) skip the boundary on that side."""
    if keyword == "c":
        # Match 'c' but not 'c++' or 'c#'
        pattern = r'\bc\b(?![+#])'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    # Escape regex special chars in the keyword
    pattern = re.escape(keyword)
    # Check if keyword starts with a word character to apply word boundary
    start_boundary = r'\b' if keyword[0].isalnum() or keyword[0] == '_' else ''
    # Check if keyword ends with a word character to apply word boundary
    end_boundary = r'\b' if keyword[-1].isalnum() or keyword[-1] == '_' else ''
    
    full_pattern = start_boundary + pattern + end_boundary
    return bool(re.search(full_pattern, text, re.IGNORECASE))

def _get_codex_context(chat_history):
    """Look at the latest user message and, for up to two matching languages/tools in _CODEX_KEYWORD_MAP, load data/codex_library/<name>.html as text. Returns a 'Codex Reference Context' block, or '' if nothing matched or the files are missing/unreadable."""
    last_user_msg = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return ""

    matched_filenames = []
    for kw, filename in _CODEX_KEYWORD_MAP.items():
        if matches_keyword(last_user_msg, kw):
            if filename not in matched_filenames:
                matched_filenames.append(filename)
            # Cap at two pages to bound how much reference text is injected into the prompt.
            if len(matched_filenames) >= 2:
                break

    if not matched_filenames:
        return ""

    context_parts = []
    library_dir = "data/codex_library"
    for filename in matched_filenames:
        filepath = os.path.join(library_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                stripped = strip_html_tags(content)
                topic = os.path.splitext(filename)[0]
                context_parts.append(f"[{topic}]\n{stripped}")
            except Exception as e:
                logger.warning(f"Error reading {filename}: {e}")

    if not context_parts:
        return ""

    return "\n\nCodex Reference Context:\n" + "\n\n".join(context_parts)


def _apply_vocab_leak_bypass(text: str) -> str:
    """
    Strip known hard-leaking special tokens from *text* to prevent injecting
    <unk> tokens into the model's context.  Only called when an adapter is
    active and vocab_leak_tokens() is non-empty.

    User-typed message content is intentionally excluded from this function;
    we only sanitize the structural prompt scaffolding (system prompt, pre-seeds).
    """
    # Token texts that tokenise to <unk> under the active adapter; injecting them would corrupt the
    # context.
    leak_map = ModelLoader.vocab_leak_tokens()
    if not leak_map:
        return text
    for token_text in leak_map:
        if token_text in text:
            logger.debug(
                "Vocab leak bypass: stripping '%s' from prompt scaffold "
                "(maps to <unk> under active adapter)",
                token_text,
            )
            text = text.replace(token_text, "")
    return text


def _truthy(value) -> bool:
    """Lenient boolean parse for config values that may arrive as bool, string ('1','true','yes','on') or anything else."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _machine_speak_enabled() -> bool:
    """Resolve the Machine Speak toggle. Per-run ACTIVE_HYPERPARAMS win (either key spelling), then the saved UI config; any config error means disabled."""
    if "machine_speak_enabled" in ACTIVE_HYPERPARAMS:
        return _truthy(ACTIVE_HYPERPARAMS.get("machine_speak_enabled"))
    if "enable_machine_speak" in ACTIVE_HYPERPARAMS:
        return _truthy(ACTIVE_HYPERPARAMS.get("enable_machine_speak"))
    try:
        return _truthy(config_store.get_ui_config().get("enable_machine_speak", False))
    except Exception:
        return False


def _machine_speak_history(system_prompt: str, chat_history: list[dict]) -> list[dict]:
    """Re-serialize each chat turn through compact_trace_for_ai (token-dense 'Machine Speak'). The system prompt is attached to the first turn only so it is encoded once."""
    compressed: list[dict] = []
    for index, msg in enumerate(chat_history):
        role = msg.get("role", "user")
        trace_turn = {
            "compiled_prompt": msg.get("content", "") if role == "user" else "",
            "response": msg.get("content", "") if role == "assistant" else "",
            "system_prompt": system_prompt if index == 0 else "",
        }
        compressed.append({**msg, "content": compact_trace_for_ai(trace_turn)})
    return compressed


def build_prompt(system_prompt, chat_history):
    """
    Builds the native prompt for distilled models based on loaded architecture.

    When NO adapter is active (base model):
        - Use the full system_prompt passed in from the Workbench.
        - Pre-seed <think>\n so the model immediately enters reasoning mode.
          Without this, the base model outputs garbled text.

    When an adapter IS active:
        - Use a minimal system prompt that matches the adapter's training data.
          The full Karl system prompt confuses adapters trained on simpler prompts.
        - Do NOT pre-seed <think>\n. The adapter generates its own <think>
          block from scratch if it wants one.
    """
    active_adapter = getattr(ModelLoader, "_active_adapter", None)
    adapter_active = bool(active_adapter)

    # When an adapter is active, we check if the user provided a custom system prompt
    # or if there is a RAG context. If the system prompt is empty or is one of the
    # default system prompts, we use the adapter's trained system prompt (_ADAPTER_SYSTEM_PROMPT).
    # Otherwise, we use the custom system prompt to let it affect the model.
    # We also always preserve any RAG context.
    
    rag_context = ""
    clean_sys = system_prompt
    # RAG context is appended to the system prompt under this marker; split it off so the default-
    # prompt comparison ignores it and it can be re-attached afterwards.
    if "\n\nRetrieved Context:\n" in system_prompt:
        parts = system_prompt.split("\n\nRetrieved Context:\n", 1)
        clean_sys = parts[0]
        rag_context = "\n\nRetrieved Context:\n" + parts[1]

    # Include legacy defaults so old saved prompts still count as 'default' after prompt upgrades.
    default_sys_prompts = LEGACY_DEFAULT_SYSTEM_PROMPTS | {DEFAULT_SYSTEM_PROMPT}

    # Whitespace-normalised comparison so formatting-only differences still count as the default
    # prompt.
    clean_sys_stripped = " ".join(clean_sys.strip().split())
    is_default = any(
        clean_sys_stripped == " ".join(p.strip().split()) 
        for p in default_sys_prompts
    )

    def is_greeting(text: str) -> bool:
        """True for bare greetings or very short (<=4 words) messages that start with one; such turns get the lightweight greeting prompt and no forced <think> seed."""
        t = text.strip().lower().rstrip(".,!?")
        if not t:
            return False
        # Direct matches
        if t in {"hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there", "hey there"}:
            return True
        # Check if it starts with a greeting and is very short (e.g. <= 4 words)
        words = t.split()
        if len(words) <= 4 and words:
            first = words[0]
            if first in {"hi", "hello", "hey", "yo", "greetings"}:
                return True
        return False

    # Get dynamic Codex reference context based on keyword matching
    codex_context = _get_codex_context(chat_history)

    # Determine if the last user message is a greeting
    last_user_msg = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    is_user_greeting = is_greeting(last_user_msg)

    if adapter_active:
        if is_default:
            effective_system = _ADAPTER_SYSTEM_PROMPT + rag_context + codex_context
        else:
            effective_system = clean_sys + rag_context + codex_context
    else:
        if is_default and is_user_greeting:
            effective_system = GREETING_SYSTEM_PROMPT + rag_context + codex_context
        elif is_default:
            effective_system = _BASE_SYSTEM_PROMPT + rag_context + codex_context
        else:
            effective_system = clean_sys + rag_context + codex_context

    # Always end the system prompt with the recency instruction, but never duplicate it.
    if _RECENCY_INSTRUCTION not in effective_system:
        effective_system = (effective_system + "\n" if effective_system else "") + _RECENCY_INSTRUCTION

    # Swarm agent prompts are detected by their role text; they have their own format so Machine Speak
    # is skipped for them.
    is_swarm = False
    if isinstance(system_prompt, str):
        if "You are an expert software engineer" in system_prompt or "Coder Agent" in system_prompt or "Architect Agent" in system_prompt:
            is_swarm = True

    machine_speak_enabled = _machine_speak_enabled() and not is_swarm
    render_history = chat_history
    if machine_speak_enabled:
        if _MACHINE_SPEAK_RULE not in effective_system:
            effective_system = (effective_system + "\n\n" if effective_system else "") + _MACHINE_SPEAK_RULE
        render_history = _machine_speak_history(effective_system, chat_history)

    # ── Vocab leak bypass: sanitise the system prompt scaffold ───────────────
    # Strip any token texts that were detected as <unk> collisions by the
    # Vocab Leak Inspector at load time.  User message content is not altered.
    if adapter_active:
        effective_system = _apply_vocab_leak_bypass(effective_system)
    # ─────────────────────────────────────────────────────────────────────────

    # Pre-seeding <think> is skipped for greetings and when the adapter's vocabulary cannot represent
    # the tag.
    preseed_think = True
    if is_user_greeting:
        preseed_think = False
    elif adapter_active and "<think>" in ModelLoader.vocab_leak_tokens():
        preseed_think = False
        logger.debug(
            "Vocab leak bypass: suppressing <think> preseed "
            "(<think> maps to <unk> under active adapter '%s')",
            active_adapter,
        )

    model_name = ModelLoader.model_name().lower()

    # Pick the chat template from the model name; unknown names use the DeepSeek-native fallback
    # below.
    if "llama" in model_name:
        # Llama 3 chat template format
        prompt = ""
        if effective_system:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{effective_system}<|eot_id|>"
        for msg in render_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if preseed_think:
            prompt += "<think>\n"

    elif "qwen" in model_name or "1.5b" in model_name or "7b" in model_name or "14b" in model_name:
        # ChatML format (Qwen-based distilled models)
        prompt = ""
        if effective_system:
            prompt += f"<|im_start|>system\n{effective_system}<|im_end|>\n"
        for msg in render_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        if preseed_think:
            prompt += "<think>\n"

    else:
        # Native DeepSeek format fallback
        prompt = ""
        if effective_system:
            prompt += f"{effective_system}"
        for msg in render_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<｜User｜>{content}"
            elif role == "assistant":
                prompt += f"<｜Assistant｜>{content}<｜end\u2581of\u2581sentence｜>"

        if preseed_think:
            prompt += "<｜Assistant｜><think>\n"
        else:
            prompt += "<｜Assistant｜>"

    return prompt
