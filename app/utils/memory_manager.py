# Program: karl-offline-lab - memory_manager.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Saves and restores the active chat session (autosave checkpoint) and the recent swarm history through the session repository.

"""Autosave checkpoint and swarm history persistence for the chat workspace."""

# Standard-library and repository imports.
import os
import re
from datetime import datetime
from app.repository.session_repository import SessionRepository

# Matches <think>...</think> reasoning blocks so they can be stripped before saving.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── Memory manager ───────────────────────────────────────────────────────
class MemoryManager:
    """Persists autosave checkpoints and swarm history, cleaning reasoning text out of saved messages."""
    def __init__(self, sessions_dir="data/sessions", repository=None):
        """Store the sessions folder and create a session repository for it unless one is supplied."""
        # Where session files live and the fixed name of the crash-recovery checkpoint.
        self.sessions_dir = sessions_dir
        self.autosave_filename = "autosave_active.json"
        # Use the supplied repository (useful for tests) or build the default one.
        self.repository = repository or SessionRepository(sessions_dir)

    @property
    def autosave_path(self) -> str:
        """Full path of the autosave checkpoint file."""
        return os.path.join(self.sessions_dir, self.autosave_filename)

    def _serialize_history(self, chat_history):
        # Imported inside the method rather than at module level.
        """Convert a session tree or flat message list into plain, saveable data with <think> blocks removed from assistant text."""
        from app.utils.session_tree import SessionTree, SessionNode
        
        # Recursively turn one tree node into a dict, cleaning assistant text on the way.
        def _clean_node_dict(node: SessionNode) -> dict:
            """Return one node as a plain dict (with assistant reasoning removed) including its children."""
            content = node.content
            if node.role == "assistant":
                # Remove hidden reasoning so it is not written to disk.
                content = _THINK_RE.sub("", content).strip()
            return {
                "id": node.id,
                "role": node.role,
                "content": content,
                "thought": node.thought,
                "attachments": getattr(node, "attachments", []),
                "children": [_clean_node_dict(c) for c in node.children]
            }

        if isinstance(chat_history, SessionTree):
            return {
            # A session tree is saved as its root node plus the id of the currently selected node.
                "root": _clean_node_dict(chat_history.root),
                "current_id": chat_history.current_id
            }

        # Otherwise treat it as a flat list of message dicts (None becomes empty).
        cleaned = []
        for msg in chat_history or []:
            if msg.get("role") == "assistant":
                content = _THINK_RE.sub("", msg.get("content", "")).strip()
                cleaned.append({**msg, "content": content})
            else:
                cleaned.append(msg)
        return cleaned

    def save_autosave_checkpoint(
        self,
        session_tree,
        active_workspace,
        model_settings: dict | None = None,
        active_tab_state: dict | None = None,
    ) -> str:
        """Atomically persist the active workspace and session tree for crash recovery."""
        # One timestamp is reused for both time fields so they always match.
        now = datetime.now().isoformat()
        payload = {
            "checkpoint_type": "autosave_active",
            "timestamp": now,
            "updated_time": now,
            "active_workspace": active_workspace,
            "active_tab_state": active_tab_state or {},
            "model_settings": model_settings or {},
            "session_tree": self._serialize_history(session_tree),
        }
        # The repository handles the actual write.
        self.repository.save(self.autosave_filename, payload)
        return self.autosave_path

    def load_autosave_checkpoint(self) -> dict | None:
        """Return the saved autosave checkpoint, or None if there is none."""
        return self.repository.get(self.autosave_filename)

    def clear_autosave_checkpoint(self):
        """Delete the autosave checkpoint and any leftover temporary file."""
        self.repository.delete(self.autosave_filename)
        self.repository.delete(self.autosave_filename + ".tmp")

    def load_swarm_history(self):
        """Return the saved swarm history list, or an empty list if missing or not a list."""
        history = self.repository.get("../swarm_history.json")
        return history if isinstance(history, list) else []

    def save_swarm_history(self, history):
        """Save only the first 10 entries of the swarm history list."""
        self.repository.save("../swarm_history.json", history[:10])

