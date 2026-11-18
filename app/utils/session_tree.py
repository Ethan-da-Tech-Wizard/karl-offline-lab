# Program: karl-offline-lab - session_tree.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Models a chat conversation as a branching tree of messages, with navigation helpers and save/load of sessions as JSON files.

"""Conversation tree model: message nodes with branching replies, a current-node cursor and JSON session save/load."""

from __future__ import annotations

# Standard-library imports for regex, ids and the stats dataclass.
import re
import uuid
from dataclasses import dataclass

# Matches <think>...</think> reasoning blocks so they can be removed before saving.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thoughts(node_dict: dict) -> dict:
    """Recursively strip reasoning content from a SessionNode dict before it's
    written to disk. AGENTS.md documents that saved sessions never persist the
    model's raw <think> trace (see memory_manager._serialize_history) -- this
    is the SessionTree-side equivalent, since SessionTree.save() is the path
    actual sessions are written through, not MemoryManager.
    """
    if node_dict.get("role") == "assistant":
        node_dict["content"] = _THINK_RE.sub("", node_dict.get("content") or "").strip()
        node_dict["thought"] = None
    for child in node_dict.get("children", []):
        _strip_thoughts(child)
    return node_dict


# ── Tree statistics ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class SessionTreeStats:
    """Read-only summary of a tree: node counts, leaf count and deepest level."""
    total_nodes: int
    message_nodes: int
    leaf_count: int
    max_depth: int

# ── Session node ─────────────────────────────────────────────────────────
class SessionNode:
    """One message in the conversation, linked to its parent and any alternative replies (children)."""
    def __init__(
        self,
        role: str,
        content: str,
        node_id: str = None,
        children: list = None,
        parent: 'SessionNode' = None,
        thought: str = None,
        attachments: list | None = None,
    ):
        """Create a message node; id and lists are filled in with defaults when not given."""
        self.role = role
        self.content = content
        # Use the given id, or make a random unique one.
        self.id = node_id or str(uuid.uuid4())
        # Fresh lists are created when none are passed so nodes never share a list.
        self.children = children or []
        self.parent = parent
        self.thought = thought
        self.attachments = attachments or []

    def add_child(
        self,
        role: str,
        content: str,
        node_id: str = None,
        attachments: list | None = None,
        thought: str = None,
    ) -> 'SessionNode':
        """Create a child message node under this one, link it, and return it."""
        # The new node points back to this node as its parent.
        child = SessionNode(role, content, node_id=node_id, parent=self, attachments=attachments, thought=thought)
        self.children.append(child)
        return child

    def to_dict(self) -> dict:
        """Convert this node and all its descendants into nested plain dicts for JSON."""
        data = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "thought": self.thought,
            "children": [c.to_dict() for c in self.children]
        }
        # Attachments are only written when present, keeping saved files small.
        if self.attachments:
            data["attachments"] = self.attachments
        return data

    @classmethod
    def from_dict(cls, data: dict, parent: 'SessionNode' = None) -> 'SessionNode':
        """Rebuild a node (and its children, recursively) from a dict made by to_dict."""
        node = cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            node_id=data.get("id"),
            thought=data.get("thought"),
            attachments=data.get("attachments", []),
            parent=parent
        )
        # Children are rebuilt recursively and linked back to this node.
        node.children = [cls.from_dict(c, parent=node) for c in data.get("children", [])]
        return node


# ── Session tree ─────────────────────────────────────────────────────────
class SessionTree:
    """A conversation tree with a "current" node; the path from the root to it is the active chat."""
    def __init__(self, root: SessionNode = None, current_node_id: str = None):
        """Start with the given root (or a new empty root) and index every node by id."""
        if root is None:
            # No tree supplied: make an empty placeholder root.
            self.root = SessionNode(role="system", content="Root", node_id="root")
            self.current_id = self.root.id
        else:
            self.root = root
            self.current_id = current_node_id or self.root.id
        
        # Quick lookup table from node id to node.
        self.nodes_map = {}
        self._rebuild_maps()

    def _rebuild_maps(self):
        """Rebuild the id lookup table and repair parent links by walking the whole tree."""
        self.nodes_map = {}
        def _walk(node):
            self.nodes_map[node.id] = node
            for child in node.children:
                child.parent = node
                _walk(child)
        _walk(self.root)

    def get_node(self, node_id: str) -> SessionNode | None:
        """Return the node with this id, or None if unknown."""
        return self.nodes_map.get(node_id)

    @property
    def current_node(self) -> SessionNode:
        """The node currently selected (falls back to the root if the id is unknown)."""
        if self.current_id not in self.nodes_map:
            # Fallback if map out of sync
            self._rebuild_maps()
        return self.nodes_map.get(self.current_id, self.root)

    def add_message(self, role: str, content: str, attachments: list | None = None, thought: str = None) -> SessionNode:
        """Append a new message under the current node and make it the current node."""
        parent = self.current_node
        child = parent.add_child(role, content, attachments=attachments, thought=thought)
        self.nodes_map[child.id] = child
        self.current_id = child.id
        return child

    def branch_from(
        self,
        node_id: str,
        role: str | None = None,
        content: str = "",
        attachments: list | None = None,
        thought: str = None,
    ) -> SessionNode | None:
        """Move to node_id and optionally append a new child as a fresh branch."""
        if not self.set_current_node(node_id):
            return None
        if role is None:
            return self.current_node
        return self.add_message(role, content, attachments=attachments, thought=thought)

    def get_active_path(self) -> list[SessionNode]:
        """Messages from the root down to the current node, excluding the placeholder root."""
        path = []
        curr = self.current_node
        # Walk upward via parent links, then reverse to get root-to-leaf order.
        while curr is not None:
            if curr.id != self.root.id:  # Skip the dummy root
                path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def get_active_path_dicts(self) -> list[dict]:
        """Active path as simple dicts (role, content, id and attachments if any) for use as a chat list."""
        items = []
        for n in self.get_active_path():
            item = {"role": n.role, "content": n.content, "id": n.id}
            if n.attachments:
                item["attachments"] = n.attachments
            items.append(item)
        return items

    def set_current_node(self, node_id: str) -> bool:
        """Select the node with this id; returns False if it does not exist."""
        # Make sure node_id exists, rebuilding maps if needed
        if node_id not in self.nodes_map:
            self._rebuild_maps()
        if node_id in self.nodes_map:
            self.current_id = node_id
            return True
        return False

    def update_current_node_content(self, text: str):
        """Replace the text of the current node (for example while a reply streams in)."""
        self.current_node.content = text

    def update_node_content(self, node_id: str, text: str) -> bool:
        """Replace the text of the node with this id; returns False if it is not found."""
        node = self.get_node(node_id)
        if node is None:
            return False
        node.content = text
        return True

    def node_depth(self, node_id: str | None = None) -> int:
        """Number of steps from the root to the given node (default: current), 0 for the root."""
        node = self.get_node(node_id or self.current_id)
        depth = 0
        while node is not None and node.id != self.root.id:
            depth += 1
            node = node.parent
        return depth

    def leaf_nodes(self) -> list[SessionNode]:
        """Return every message node that has no replies under it."""
        leaves: list[SessionNode] = []

        # Visit every node; a node with no non-root children is a leaf.
        def _walk(node: SessionNode):
            message_children = [child for child in node.children if child.id != self.root.id]
            if node.id != self.root.id and not message_children:
                leaves.append(node)
            for child in node.children:
                _walk(child)

        _walk(self.root)
        return leaves

    def stats(self) -> SessionTreeStats:
        """Compute node count, message count, leaf count and maximum depth for the whole tree."""
        total = 0
        message_nodes = 0
        max_depth = 0

        # Counts are updated in the enclosing function as the walk descends.
        def _walk(node: SessionNode, depth: int):
            nonlocal total, message_nodes, max_depth
            total += 1
            if node.id != self.root.id:
                message_nodes += 1
                max_depth = max(max_depth, depth)
            for child in node.children:
                _walk(child, depth + 1)

        _walk(self.root, 0)
        return SessionTreeStats(
            total_nodes=total,
            message_nodes=message_nodes,
            leaf_count=len(self.leaf_nodes()),
            max_depth=max_depth,
        )

    def active_branch_label(self) -> str:
        """Short label for the active branch: the last message role plus the first 8 characters of its id."""
        path = self.get_active_path()
        if not path:
            return "root"
        leaf = path[-1]
        return f"{leaf.role}:{leaf.id[:8]}"

    def to_dict(self) -> dict:
        """Convert the whole tree plus the current node id into a plain dict."""
        return {
            "root": self.root.to_dict(),
            "current_id": self.current_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SessionTree':
        """Build a tree from saved data; empty or invalid data gives a fresh empty tree."""
        if not data or "root" not in data:
            return cls()
        root = SessionNode.from_dict(data["root"])
        return cls(root, data.get("current_id"))

    def clear(self):
        """Reset to an empty tree with a new placeholder root."""
        self.root = SessionNode(role="system", content="Root", node_id="root")
        self.current_id = self.root.id
        self.nodes_map = {self.root.id: self.root}

    def copy(self) -> 'SessionTree':
        """Return an independent deep copy made by round-tripping through a dict."""
        return SessionTree.from_dict(self.to_dict())

    # List interface duck-typing helpers: these let the tree behave like a list of the active messages.
    def __len__(self):
        """Number of messages on the active path."""
        return len(self.get_active_path())

    def __getitem__(self, index):
        """Index or slice into the active path (as message dicts)."""
        active = self.get_active_path_dicts()
        return active[index]

    def __iter__(self):
        """Iterate over the active path as message dicts."""
        return iter(self.get_active_path_dicts())

    def __bool__(self):
        """True when the active path has at least one message."""
        return len(self.get_active_path()) > 0

    def append(self, msg: dict):
        """Add a message dict (needs role and content keys) under the current node."""
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise TypeError("Appended message must be a dict containing 'role' and 'content'")
        self.add_message(msg["role"], msg["content"], attachments=msg.get("attachments"))

    # Folder where session files are written, relative to the working directory.
    SESSIONS_DIR = "data/sessions"

    def save(self, session_id: str | None = None) -> str:
        """Write this tree to data/sessions/{session_id}.json. Returns the path."""
        import os
        import json
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]
        path = os.path.join(self.SESSIONS_DIR, f"{session_id}.json")
        payload = self.to_dict()
        # Remove stored reasoning text so it never reaches disk.
        _strip_thoughts(payload["root"])
        # Record the id inside the file so it can be listed later.
        payload["session_id"] = session_id
        # Store first user message as preview
        for node_data in payload.get("root", {}).get("children", []):
            if node_data.get("role") == "user":
                payload["preview"] = node_data.get("content", "")[:80]
                break
        import os as _os
        # Write to a temporary file first, then rename, so a crash never leaves a half-written session.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> tuple['SessionTree', str]:
        """Load a session from disk. Returns (tree, session_id)."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session_id = data.get("session_id", "unknown")
        tree = cls.from_dict(data)
        return tree, session_id

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """Return metadata for all saved sessions, newest first."""
        import os
        import json
        if not os.path.exists(cls.SESSIONS_DIR):
            return []
        sessions = []
        # Look at each saved session file; skip non-JSON files and unreadable ones.
        for fname in sorted(os.listdir(cls.SESSIONS_DIR), reverse=True):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(cls.SESSIONS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "path": path,
                    "session_id": data.get("session_id", fname[:-5]),
                    "preview": data.get("preview", "(empty)"),
                    "mtime": os.path.getmtime(path),
                })
            except Exception:
                pass
        # Newest first, by file modification time.
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return sessions

    def delete_session_file(self, path: str):
        """Delete a saved session file if it exists."""
        import os
        if os.path.exists(path):
            os.remove(path)
