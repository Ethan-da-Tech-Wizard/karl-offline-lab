# ==============================================================================
# Program: swarm_agents.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Multi-agent persona definitions and specialized agent roles.
#          Configures Architect, Developer, Reviewer, and Security Analyst
#          agents for collaborative task solving.
# ==============================================================================

"""
Swarm Agents — Karl Workbench
==============================
Defines the specialized ArchitectAgent, CoderAgent, and TesterAgent subclasses
orchestrating local LLM calls and system prompts to perform codebase tasks.
"""

import os
import re
import json
import logging
import subprocess
from typing import Dict, Any, Optional, Callable
from app.engine.model_loader import ModelLoader
from app.engine.agent_memory import CodebaseMemory, keywords_from_task
from core.interaction_loop import build_prompt
from core.default_prompts import SWARM_ARCHITECT_SYSTEM_PROMPT, SWARM_CODER_SYSTEM_PROMPT

logger = logging.getLogger("karl.swarm_agents")


# Single message returned to the model whenever a tool is refused for leaving the workspace.
_SECURITY_BLOCK_MSG = "ERROR: Security Block: Path traversal outside workspace boundary is prohibited."


def _safe_workspace_path(workspace_path: str, rel: str) -> str | None:
    """
    Returns the fully-resolved absolute path for *rel* inside *workspace_path*,
    or None if the path escapes the workspace via any mechanism (parent traversal,
    absolute injection, or symlink escape).

    Symlink escape example:
        workspace/evil -> /etc/passwd
        os.path.realpath(workspace/evil) == /etc/passwd  → blocked
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    # An empty path would resolve to the workspace root itself; treat it as invalid.
    if not rel:
        return None
    # realpath resolves symlinks on BOTH sides, so a link inside the workspace pointing outside is caught.
    ws_real = os.path.realpath(workspace_path)
    target = os.path.realpath(os.path.join(ws_real, rel))
    # The trailing os.sep stops "/work/foo" from matching a sibling like "/work/foobar".
    if target == ws_real or target.startswith(ws_real + os.sep):
        return target
    return None


# ── Tool Registry ─────────────────────────────────────────────────────────────
# Tools are registered here so both CoderAgent and external callers can extend
# the dispatch table. Each value is (executor_fn, description_for_prompt).

# name -> (function(workspace_path, args) -> str, prompt description). Filled by @register_tool.
_TOOL_REGISTRY: dict[str, tuple[Callable, str]] = {}

def register_tool(name: str, description: str):
    """Decorator to register a tool function in the global tool registry."""
    # Step 1: Validate input parameters, operational preconditions, and state
    def decorator(fn):
        """Store ``fn`` under the tool name and return it unchanged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        _TOOL_REGISTRY[name] = (fn, description)
        return fn
    return decorator

def get_tool_schema_block(allowed_tools: set[str] | None = None) -> str:
    """Returns the tool schema XML block to inject into coder prompts."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lines = ["<tools>"]
    # Skip tools the current agent profile is not allowed to use, so the model never sees them.
    for name, (_, desc) in _TOOL_REGISTRY.items():
        # Iterate over items, evaluate loop invariant, and update progress
        if allowed_tools is not None and name not in allowed_tools:
            continue
        lines.append(f"  <tool name='{name}'>{desc}</tool>")
    lines.append("</tools>")
    return "\n".join(lines)


@register_tool("write_file", "write_file(path, content) — overwrite a workspace file. path is relative to workspace root.")
def _tool_write_file(workspace_path: str, args: dict) -> str:
    """Tool: overwrite a file inside the workspace and report the line count.

    Markdown fences around the content are removed (models often add them), and
    paths outside the workspace are refused.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    import pathlib
    rel = args.get("path", "")
    content = args.get("content", "")
    
    # Drop an opening ```lang line and a closing ``` so only the code is saved.
    # Strip markdown code blocks if the model wrapped the content in them
    content = content.strip()
    if content.startswith("```"):
        first_nl = content.find("\n")
        if first_nl != -1:
            content = content[first_nl:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
            
    target = _safe_workspace_path(workspace_path, rel)
    if target is None:
        logger.warning("SECURITY ALERT: path traversal blocked in write_file. rel=%r workspace=%r", rel, workspace_path)
        return _SECURITY_BLOCK_MSG
    full = pathlib.Path(target)
    # Create missing parent folders so the model can add new files.
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content.splitlines())} lines to {rel}"

@register_tool("read_file", "read_file(path) — read current contents of a workspace file.")
def _tool_read_file(workspace_path: str, args: dict) -> str:
    """Tool: return up to 6000 characters of a workspace file (or an ERROR string)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    import pathlib
    rel = args.get("path", "")
    target = _safe_workspace_path(workspace_path, rel)
    if target is None:
        logger.warning("SECURITY ALERT: path traversal blocked in read_file. rel=%r workspace=%r", rel, workspace_path)
        return _SECURITY_BLOCK_MSG
    full = pathlib.Path(target)
    if not full.exists():
        return f"ERROR: file not found: {rel}"
    try:
        # Execute guarded operations and establish recovery boundaries
        return full.read_text(encoding="utf-8", errors="ignore")[:6000]
    except Exception as e:
        return f"ERROR: {e}"

@register_tool("grep_workspace", "grep_workspace(pattern) — find lines matching regex pattern across all .py files in workspace.")
def _tool_grep_workspace(workspace_path: str, args: dict) -> str:
    """Tool: regex search over workspace .py files; at most 40 matches are returned."""
    # Step 1: Validate input parameters, operational preconditions, and state
    import pathlib
    import re as _re
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: pattern required"
    try:
        # Execute guarded operations and establish recovery boundaries
        compiled = _re.compile(pattern)
    except _re.error as e:
        return f"ERROR: invalid regex: {e}"
    results = []
    for f in pathlib.Path(workspace_path).rglob("*.py"):
        # Iterate over items, evaluate loop invariant, and update progress
        # Skip version-control, virtualenv and cache folders: they are huge and irrelevant.
        if ".git" in f.parts or "venv" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            # Execute guarded operations and establish recovery boundaries
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                # Iterate over items, evaluate loop invariant, and update progress
                if compiled.search(line):
                    results.append(f"{f.relative_to(workspace_path)}:{i}: {line.rstrip()}")
                    # Cap output to keep the tool result small in the model's context.
                    if len(results) >= 40:
                        break
        except Exception:
            pass
        # The inner break only leaves the per-file loop; stop scanning further files too.
        if len(results) >= 40:
            break
    return "\n".join(results) if results else "No matches found."

@register_tool("shell_run", "shell_run(command) — run a shell command in the workspace directory. Returns stdout+stderr. Timeout 15s.")
def _tool_shell_run(workspace_path: str, args: dict) -> str:
    """Tool: run an arbitrary shell command in the workspace (15s timeout, output cut to 3000 chars).

    WARNING: shell=True with model-written text and no sandbox; the command can
    access anything the user can.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    cmd = args.get("command", "")
    if not cmd:
        return "ERROR: command required"
    try:
        # Execute guarded operations and establish recovery boundaries
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=workspace_path, timeout=15
        )
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out (15s)"
    except Exception as e:
        return f"ERROR: {e}"

@register_tool("lint_python", "lint_python(path) — run pyflakes on a Python file and return violations.")
def _tool_lint_python(workspace_path: str, args: dict) -> str:
    """Tool: run pyflakes on one workspace file and return its output."""
    # Step 1: Validate input parameters, operational preconditions, and state
    import pathlib
    rel = args.get("path", "")
    target = _safe_workspace_path(workspace_path, rel)
    if target is None:
        logger.warning("SECURITY ALERT: path traversal blocked in lint_python. rel=%r workspace=%r", rel, workspace_path)
        return _SECURITY_BLOCK_MSG
    full = pathlib.Path(target)
    if not full.exists():
        return f"ERROR: file not found: {rel}"
    try:
        # Execute guarded operations and establish recovery boundaries
        result = subprocess.run(
            ["python", "-m", "pyflakes", str(full)],
            capture_output=True, text=True, timeout=10
        )
        out = (result.stdout + result.stderr).strip()
        return out if out else "No violations found."
    except Exception as e:
        return f"ERROR: {e}"


class BaseSwarmAgent:
    """Shared plumbing for agents that call the local LLM with a fixed system prompt."""
    def __init__(self, system_prompt: str, temperature: float = 0.2, max_tokens: int = 2048):
        """Store the system prompt and default sampling settings."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def clean_output(self, raw: str) -> str:
        """Strips DeepSeek reasoning tags and markdown fences from the output."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        cleaned = raw.strip()
        # Clean markdown code block fences (e.g. ```json ... ```)
        cleaned = re.sub(r"```(?:[a-zA-Z0-9+#-]+)?\s*", "", cleaned).replace("```", "").strip()
        return cleaned

    def call_llm(self, user_prompt: str) -> str:
        """Invokes the local LLM singleton in a thread-safe manner."""
        # Step 1: Validate input parameters, operational preconditions, and state
        llm = ModelLoader.get_instance()
        history = [{"role": "user", "content": user_prompt}]
        prompt = build_prompt(self.system_prompt, history)

        response = llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            # Stop at ChatML boundaries so the model does not invent the next turn.
            stop=["<|im_end|>", "<|im_start|>"],
            repeat_penalty=1.1,
        )
        return response["choices"][0]["text"]


# ── Architect Agent ───────────────────────────────────────────────────────────

class ArchitectAgent(BaseSwarmAgent):
    """Planner: turns an objective plus file contents into a JSON task plan."""
    SYSTEM_PROMPT = SWARM_ARCHITECT_SYSTEM_PROMPT

    def __init__(self):
        """Low temperature (0.1) for stable, parseable JSON."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(self.SYSTEM_PROMPT, temperature=0.1, max_tokens=1536)

    def create_plan(self, objective: str, files_context: Dict[str, str]) -> Dict[str, Any]:
        """Ask the model for a JSON plan {"explanation", "tasks": [...]}.

        If the reply is not valid JSON, return an empty task list with the raw text in the explanation.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        context_str = ""
        for path, content in files_context.items():
            # Iterate over items, evaluate loop invariant, and update progress
            context_str += f"=== FILE: {path} ===\n{content}\n\n"

        prompt = (
            f"Objective: {objective}\n\n"
            f"Here is the context of the files:\n{context_str}\n"
            f"Produce your JSON plan now."
        )

        raw = self.call_llm(prompt)
        cleaned = self.clean_output(raw)
        
        try:
            # Execute guarded operations and establish recovery boundaries
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Never raise on bad model output; the orchestrator can handle an empty plan.
            # Fallback if model did not return valid JSON
            return {
                "explanation": f"Failed to parse JSON plan from model. Raw output: {cleaned[:200]}",
                "tasks": []
            }


def extract_python_code(raw_text: str) -> Optional[str]:
    """Extract python/code block from raw response, excluding diff blocks."""
    # Step 1: Validate input parameters, operational preconditions, and state
    # Prefer an explicit python/py block first.
    match = re.search(r"```(?:python|py)?\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Otherwise take any fenced block except a diff.
    match_any = re.search(r"```(?!\s*diff)(?:[a-zA-Z0-9+#-]+)?\n(.*?)\n```", raw_text, re.DOTALL | re.IGNORECASE)
    if match_any:
        return match_any.group(1).strip()
    return None


def apply_search_replace(current_content: str, raw_text: str) -> str:
    """Find and apply all search-replace blocks (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE)"""
    # Step 1: Validate input parameters, operational preconditions, and state
    # Aider-style blocks: <<<<<<< SEARCH / ======= / >>>>>>> REPLACE.
    blocks = re.findall(
        r"<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE",
        raw_text,
        re.DOTALL
    )
    if not blocks:
        return current_content

    new_content = current_content
    for search, replace in blocks:
        # Iterate over items, evaluate loop invariant, and update progress
        search_str = search.strip()
        replace_str = replace.strip()
        # An empty SEARCH means "append" (or "create" when the file is empty).
        if not search_str:
            if not new_content.strip():
                new_content = replace_str
            else:
                new_content = new_content + "\n\n" + replace_str
        else:
            if search_str in new_content:
                new_content = new_content.replace(search_str, replace_str)
            else:
                if not new_content.strip():
                    new_content = replace_str
                else:
                    # Behaves like the exact branch above (search was already stripped); kept as a defensive path.
                    # Loose whitespace fallback
                    new_content = new_content.replace(search.strip(), replace_str)
    return new_content


def parse_reasoning_and_tool(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parses agent output to enforce that any <tool_call> is preceded by <reasoning>.
    Returns (reasoning_content, tool_content).
    
    If <tool_call> is present but not preceded by <reasoning>, raises ValueError.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    reasoning_matches = list(re.finditer(r"<reasoning>(.*?)</reasoning>", raw_text, re.DOTALL | re.IGNORECASE))
    tool_matches = list(re.finditer(r"<tool_call(?:\s+[^>]*)?>(.*?)</tool_call>", raw_text, re.DOTALL | re.IGNORECASE))

    if tool_matches:
        first_tool_pos = tool_matches[0].start()
        has_prior_reasoning = False
        reasoning_content = None
        for rm in reasoning_matches:
            # Iterate over items, evaluate loop invariant, and update progress
            if rm.end() <= first_tool_pos:
                has_prior_reasoning = True
                reasoning_content = rm.group(1).strip()
                break
        
        if not has_prior_reasoning:
            raise ValueError(
                "Action blocked: You must write down your reasoning inside <reasoning>...</reasoning> "
                "tags before calling a tool."
            )
        
        tool_content = tool_matches[0].group(1).strip()
        return reasoning_content, tool_content

    return None, None


# ── Coder Agent ───────────────────────────────────────────────────────────────

class CoderAgent(BaseSwarmAgent):
    """Writer: edits one file by looping over tool calls (read, write, lint) until done."""
    SYSTEM_PROMPT = SWARM_CODER_SYSTEM_PROMPT

    def __init__(self):
        """Moderate temperature (0.3) and a large token budget for whole-file output."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(self.SYSTEM_PROMPT, temperature=0.3, max_tokens=4096)
        # None means every registered tool is allowed.
        self.allowed_tools: set[str] | None = None

    def generate(self, task: dict, workspace_context: dict,
                 workspace_path: str = ".",
                 token_callback: Callable[[str], None] | None = None,
                 guidance_getter: Callable[[], list[str]] | None = None,
                 memory_hint: str = "",
                 temperature_override: float | None = None,
                 allowed_paths: set[str] | None = None) -> str:
        """
        Multi-turn tool loop. Model reads workspace, thinks, writes files.
        Returns the final written content for the primary task file, or an error string.

        token_callback: if provided, called with each generated token for streaming.
        guidance_getter: if provided, called at the top of every tool-loop turn;
            any strings it returns are woven in as human-guidance messages before
            the next generation, enabling live mid-task steering.
        memory_hint: optional "known past failure patterns" block (see
            SwarmMemory.recall) injected into the system prompt so the agent
            doesn't repeat mistakes already made on this codebase.
        temperature_override: sampling temperature for this call; falls back to
            self.temperature when None (used by Multiverse candidate generation
            to explore genuinely different solutions per candidate).
        allowed_paths: when provided, write_file calls targeting any other
            workspace-relative path are blocked — keeps the coder confined to
            files the Architect actually planned to touch.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Shared model singleton; loads it if needed.
        llm = ModelLoader.get_instance()
        temperature = self.temperature if temperature_override is None else temperature_override
        tool_schema = get_tool_schema_block(self.allowed_tools)
        memory_reference = ""
        try:
            # Execute guarded operations and establish recovery boundaries
            memory = CodebaseMemory(workspace_path)
            # Re-index the workspace on every call so signatures reflect the latest files.
            memory.build_index()
            memory_reference = memory.query_memory(keywords_from_task(task))
        except Exception as exc:
            logger.debug("CodebaseMemory lookup failed: %s", exc)

        # Only the first 8 context files, truncated to 800 characters each, to protect the context window.
        # Build initial prompt
        context_snippet = "\n".join(
            f"--- {k} ---\n{v[:800]}" for k, v in list(workspace_context.items())[:8]
        )
        memory_block = ""
        if memory_reference:
            memory_block = (
                "\n\nCodebase Interfaces & Signatures Reference:\n"
                "Use the following existing codebase signatures to ensure integration:\n"
                f"{memory_reference}"
            )
        swarm_memory_block = f"\n\n{memory_hint}\nAvoid repeating these mistakes." if memory_hint else ""
        system = (
            "You are an expert software engineer. You MUST reason before acting.\n"
            "Use the tools below to read existing code, then write correct, tested changes.\n"
            "Always call read_file before writing to understand the existing implementation.\n"
            "After writing files, call lint_python and fix any violations.\n"
            "When finished with ALL changes, call done() with no arguments.\n\n"
            f"{tool_schema}\n\n"
            "Tool call format:\n"
            "<tool_call name='TOOL_NAME'>\n"
            "  param_name: value\n"
            "</tool_call>\n\n"
            # The closing form must match the parser below, which only recognizes <tool_call ...>.
            "To finish: <tool_call name='done'></tool_call>"
            f"{memory_block}"
            f"{swarm_memory_block}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Task: Edit {task['filepath']}\n\n"
                f"Instructions: {task['instructions']}\n\n"
                f"Workspace context:\n{context_snippet}\n\n"
                "IMPORTANT RULES FOR YOUR RESPONSE:\n"
                "- Write the COMPLETE new content of the file from scratch.\n"
                "- Do NOT write a unified diff (do NOT use + or - lines). Write the full valid Python file contents as you want it saved.\n"
                "- Make sure any Python code is wrapped inside a clean markdown code block: ```python ... ```"
            )}
        ]

        # Last file content the model wrote; returned as the result.
        written_content = ""

        # At most 8 model turns per task, a hard stop against endless tool loops.
        for _turn in range(8):
            # Iterate over items, evaluate loop invariant, and update progress
            # Pull any human guidance typed while the swarm runs and add it to the conversation.
            if guidance_getter:
                for guidance_msg in guidance_getter():
                    # Iterate over items, evaluate loop invariant, and update progress
                    messages.append({
                        "role": "user",
                        "content": f"Human guidance (mid-task correction): {guidance_msg}",
                    })

            # Build prompt and generate
            # Redundant: the same name is already imported at module level.
            from core.interaction_loop import build_prompt
            prompt_text = build_prompt(system, messages[1:])

            raw = ""
            # Streaming path (live tokens to the UI) with a non-streaming retry if streaming raises.
            if token_callback:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    res = llm(prompt_text, max_tokens=self.max_tokens,
                              temperature=temperature, stream=True,
                              # Stop right after a tool call closes so the tool can run before the model continues.
                              stop=["</tool_call>", "<|im_end|>", "<|im_start|>"], echo=False,
                              repeat_penalty=1.1)
                    if hasattr(res, "__iter__") and not isinstance(res, dict):
                        for chunk in res:
                            # Iterate over items, evaluate loop invariant, and update progress
                            tok = chunk["choices"][0].get("text", "")
                            raw += tok
                            token_callback(tok)
                    else:
                        if isinstance(res, dict):
                            raw = res["choices"][0]["text"]
                        else:
                            raw = str(res)
                except Exception:
                    res = llm(prompt_text, max_tokens=self.max_tokens,
                              temperature=temperature, stream=False,
                              stop=["<|im_end|>", "<|im_start|>"], echo=False,
                              repeat_penalty=1.1)
                    if isinstance(res, dict):
                        raw = res["choices"][0]["text"]
                    else:
                        raw = str(res)
            else:
                res = llm(prompt_text, max_tokens=self.max_tokens,
                           temperature=temperature, stream=False,
                           stop=["<|im_end|>", "<|im_start|>"], echo=False,
                           repeat_penalty=1.1)
                if isinstance(res, dict):
                    raw = res["choices"][0]["text"]
                else:
                    raw = str(res)
            # Keep the model's reply in history so later turns see what it already did.
            messages.append({"role": "assistant", "content": raw})
    
            # Format: <tool_call name='TOOL'>key: value lines</tool_call>.
            # Parse tool calls from raw output
            tool_calls = re.findall(
                r"<tool_call name='([^']+)'>(.*?)</tool_call>",
                raw, re.DOTALL
            )

            # No tool calls: on the first turn treat the reply itself as the answer (search/replace blocks,
            # a code block, or plain text), then stop.
            if not tool_calls:
                if _turn == 0:
                    if "<<<<<<< SEARCH" in raw:
                        existing = _tool_read_file(workspace_path, {"path": task["filepath"]})
                        file_content = "" if existing.startswith(("ERROR:", "SECURITY")) else existing
                        written_content = apply_search_replace(file_content, raw)
                    else:
                        extracted_code = extract_python_code(raw)
                        if extracted_code is not None:
                            written_content = extracted_code
                        else:
                            try:
                                # Execute guarded operations and establish recovery boundaries
                                reasoning, tool_content = parse_reasoning_and_tool(raw)
                                if tool_content is not None:
                                    written_content = tool_content
                                else:
                                    written_content = self.clean_output(raw)
                            except ValueError:
                                written_content = self.clean_output(raw)
                break

            tool_results = []
            done_called = False

            for tool_name, tool_body in tool_calls:
                # Iterate over items, evaluate loop invariant, and update progress
                if tool_name == "done":
                    done_called = True
                    break

                # Same key: value parser as tool_executor; large free-text keys take the rest of the body.
                # Robust YAML-like parser
                args = {}
                lines = tool_body.splitlines()
                i = 0
                while i < len(lines):
                    # Iterate over items, evaluate loop invariant, and update progress
                    line = lines[i]
                    match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*:(.*)$", line)
                    if match:
                        key = match.group(1).strip()
                        val = match.group(2).strip()
                        if key in ("content", "command", "code", "text", "script", "query"):
                            # The rest of the tool body belongs to this key
                            remaining_lines = lines[i:]
                            first_line = remaining_lines[0]
                            _, _, first_line_val = first_line.partition(":")
                            remaining_content = [first_line_val] + remaining_lines[1:]
                            raw_val = "\n".join(remaining_content).lstrip("\n")
                            
                            # Extract code from markdown block if present
                            raw_val = raw_val.strip()
                            match_block = re.search(r"```(?:[a-zA-Z0-9+#-]+)?\n(.*?)\n```", raw_val, re.DOTALL)
                            if match_block:
                                raw_val = match_block.group(1).strip()
                            else:
                                match_unclosed = re.search(r"```(?:[a-zA-Z0-9+#-]+)?\n(.*)$", raw_val, re.DOTALL)
                                if match_unclosed:
                                    raw_val = match_unclosed.group(1).strip()
                                    
                            args[key] = raw_val
                            break
                        else:
                            args[key] = val
                            i += 1
                    else:
                        i += 1

                # Order of checks: profile permission, then known tool, then drift guard for writes.
                if self.allowed_tools is not None and tool_name not in self.allowed_tools:
                    result_text = f"ERROR: tool '{tool_name}' is disabled for this agent profile."
                elif tool_name in _TOOL_REGISTRY:
                    executor, _ = _TOOL_REGISTRY[tool_name]
                    blocked_path = None
                    # Drift guard: only files the architect planned may be written.
                    if tool_name == "write_file" and allowed_paths is not None:
                        rel = args.get("path", "").strip().replace("\\", "/").lstrip("/")
                        if rel not in allowed_paths:
                            blocked_path = rel

                    if blocked_path is not None:
                        result_text = (
                            f"ERROR: Drift Guard: '{blocked_path}' is not part of the architect's "
                            f"task plan. Write blocked. Allowed file(s): "
                            f"{', '.join(sorted(allowed_paths)) or 'none'}."
                        )
                    else:
                        try:
                            # Execute guarded operations and establish recovery boundaries
                            result_text = executor(workspace_path, args)
                        except Exception as ex:
                            result_text = f"ERROR: tool raised exception: {ex}"

                        # Remember the last written content so it can be returned to the orchestrator.
                        # Capture written content for return value
                        if tool_name == "write_file":
                            written_content = args.get("content", "")
                else:
                    result_text = f"ERROR: unknown tool '{tool_name}'"
    
                tool_results.append(f"<tool_result name='{tool_name}'>\n{result_text}\n</tool_result>")
    
            # The model signalled completion.
            if done_called:
                break

            if tool_results:
                messages.append({"role": "user", "content": "\n".join(tool_results)})

        # Fall back to a placeholder comment so callers always get a string.
        return written_content or f"# CoderAgent completed task: {task['filepath']}"

    def edit_file(
        self,
        filepath: str,
        current_content: str,
        instructions: str,
        test_failure_trace: Optional[str] = None
    ) -> str:
        """Convenience wrapper: edit ``filepath`` with ``instructions``, optionally mentioning a previous test failure."""
        # Step 1: Validate input parameters, operational preconditions, and state
        task = {"filepath": filepath, "instructions": instructions}
        workspace_context = {filepath: current_content}
        if test_failure_trace:
            task["instructions"] += f"\nPrevious failure: {test_failure_trace}"
        return self.generate(task, workspace_context)


# ── Tester Agent ──────────────────────────────────────────────────────────────

class TesterAgent:
    """
    Programmatic Tester Agent. Runs test scripts or subprocess commands
    locally and parses exit codes and tracebacks.
    """

    def __init__(self, workspace_path: str):
        """``workspace_path`` is the working directory for test commands."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.workspace_path = workspace_path

    def run(self, command: str, workspace_path: str) -> tuple[bool, str]:
        """Runs a subprocess command and returns (passed, trace)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        res = self.run_test_command(command)
        return res["passed"], res["error_trace"] or res["output"]

    def run_test_command(self, test_cmd: str) -> Dict[str, Any]:
        """Runs a shell test command inside the workspace and captures results."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            res = subprocess.run(
                test_cmd,
                # The command comes from the task/config, not from the model, and is run through the shell.
                shell=True,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            passed = res.returncode == 0
            output = res.stdout + "\n" + res.stderr
            
            # Simple trace extractor
            error_trace = ""
            if not passed:
                # Extract Python traceback from stderr/stdout
                lines = output.split("\n")
                tb_idx = -1
                for idx, line in enumerate(lines):
                    # Iterate over items, evaluate loop invariant, and update progress
                    # Python prints exactly "Traceback (most recent call last):"; match that marker.
                    if "Traceback (most recent call last):" in line:
                        tb_idx = idx
                        break
                if tb_idx != -1:
                    error_trace = "\n".join(lines[tb_idx:])
                else:
                    # Capture the last 20 lines of log as trace
                    # Last 20 lines usually contain the failing assertion or error.
                    error_trace = "\n".join(lines[-20:])

            return {
                "passed": passed,
                "output": output,
                "error_trace": error_trace
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "output": "Test execution timed out after 300 seconds.",
                "error_trace": "TimeoutExpired"
            }
        except Exception as e:
            return {
                "passed": False,
                "output": f"Failed to execute tests: {e}",
                "error_trace": str(e)
            }
