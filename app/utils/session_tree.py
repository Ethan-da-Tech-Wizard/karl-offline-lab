# ==============================================================================
# Program: session_tree.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tree data structure governing branching conversation threads.
#          Supports non-linear dialogue exploration, backtracking, and
#          alternative prompt paths.
# ==============================================================================

"""Branching conversation tree (SessionTree/SessionNode) with JSON session persistence.

Classes:
  SessionTreeStats - Immutable summary counts for a tree (nodes, message nodes, leaves, deepest message depth).
  SessionNode - One chat message in the conversation tree; siblings are alternative branches.
  SessionTree - Branching conversation: a tree of messages with a cursor (current_id) selecting the active branch.
"""

# Role in the app: SessionTree is the data model behind Workbench conversations (see AGENTS.md "Session Branching").
# The Workbench chat view renders the active path, the branch navigator lists leaf_nodes(), MemoryManager and
# session_repository persist the tree through to_dict()/save(), and the inference threads receive the active path
# as a plain list of {"role", "content"} dicts (the list-style methods at the bottom make the tree a drop-in
# replacement for the old list-based chat_history).
# Structure: a dummy system node with id "root" holds every first message, so the tree can have several alternative
# first messages. Every other node is one chat message; siblings are alternative continuations ("branches"). A cursor
# (current_id) selects the active branch. Tests: tests/test_session_tree.py.
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

# Matches <think>...</think> reasoning blocks across lines, ignoring case.
# DOTALL lets "." cross newlines inside a reasoning block; IGNORECASE tolerates <Think> from sloppy model output.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thoughts(node_dict: dict) -> dict:
    """Recursively strip reasoning content from a SessionNode dict before it's
    written to disk. AGENTS.md documents that saved sessions never persist the
    model's raw <think> trace (see memory_manager._serialize_history) -- this
    is the SessionTree-side equivalent, since SessionTree.save() is the path
    actual sessions are written through, not MemoryManager.
    """
    # Only assistant messages carry reasoning; user content is stored exactly as typed.
    if node_dict.get("role") == "assistant":
        node_dict["content"] = _THINK_RE.sub("", node_dict.get("content") or "").strip()

        # The separate `thought` field is cleared as well, otherwise the trace would still be written via to_dict().
        node_dict["thought"] = None

    # Recursion mirrors the tree shape, so every branch (not only the active one) is sanitised.
    for child in node_dict.get("children", []):
        _strip_thoughts(child)
    return node_dict


@dataclass(frozen=True)
class SessionTreeStats:
    """Immutable summary counts for a tree (nodes, message nodes, leaves, deepest message depth).
    """
    # Field meanings: total_nodes counts the dummy root; message_nodes excludes it; leaf_count is the number of branch tips;
    # max_depth is the longest root-to-message distance (a one-message chat has depth 1).
    total_nodes: int
    message_nodes: int
    leaf_count: int
    max_depth: int

class SessionNode:
    """One chat message in the conversation tree; siblings are alternative branches."""
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
        """Create a node; a UUID is generated if no id is given."""
        self.role = role
        self.content = content

        # Caller-supplied ids are kept so loaded sessions and tests get stable references; new nodes get random UUIDs.
        self.id = node_id or str(uuid.uuid4())

        # `or []` creates a fresh list per node; a default [] argument would be shared by every node.
        self.children = children or []
        # Weak back-pointer used to walk up to the root; it is deliberately not serialized (to_dict) because
        # the tree structure already implies it, and from_dict/_rebuild_maps restore it on load.
        self.parent = parent
        self.thought = thought

        # Attachments are opaque JSON-able dicts (e.g. image references) carried with the message.
        self.attachments = attachments or []

    def add_child(
        self,
        role: str,
        content: str,
        node_id: str = None,
        attachments: list | None = None,
        thought: str = None,
    ) -> 'SessionNode':
        """Create a child node under this one, append it and return it."""
        # Passing parent=self wires the back-pointer immediately, so get_active_path() works without a rebuild.
        child = SessionNode(role, content, node_id=node_id, parent=self, attachments=attachments, thought=thought)
        self.children.append(child)
        return child

    def to_dict(self) -> dict:
        """Serialize this node and its subtree (attachments only when present)."""
        data = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "thought": self.thought,
            # Recursive; deep conversations are limited by Python's recursion limit, which chat depth never approaches.
            "children": [c.to_dict() for c in self.children]
        }
        if self.attachments:
            data["attachments"] = self.attachments
        return data

    @classmethod
    def from_dict(cls, data: dict, parent: 'SessionNode' = None) -> 'SessionNode':
        """Rebuild a node subtree from serialized data, wiring parent pointers."""
        node = cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            node_id=data.get("id"),
            thought=data.get("thought"),
            attachments=data.get("attachments", []),
            parent=parent
        )

        # Children are attached after the parent exists so each child can receive its parent pointer.
        node.children = [cls.from_dict(c, parent=node) for c in data.get("children", [])]
        return node


class SessionTree:
    """Branching conversation: a tree of messages with a cursor (current_id) selecting the active branch.

    It also behaves like a read-mostly list of the active path (len, iteration,
    indexing, append) so list-based code can use it unchanged.
    """
    def __init__(self, root: SessionNode = None, current_node_id: str = None):
        """Wrap an existing root or create an empty tree with a dummy system root."""
        # A brand-new tree gets a dummy "root" system node; a loaded tree keeps the root (and cursor) it was saved with.
        if root is None:
            self.root = SessionNode(role="system", content="Root", node_id="root")
            self.current_id = self.root.id
        else:
            self.root = root
            self.current_id = current_node_id or self.root.id
        
        # id -> node index makes cursor moves O(1) instead of searching the whole tree on every click.
        # Index built eagerly so the first set_current_node call is already O(1).
        self.nodes_map = {}
        self._rebuild_maps()

    def _rebuild_maps(self):
        """Rebuild the id->node index and re-link parent pointers by walking from the root."""
        # Reset first: nodes removed from the tree must not linger in the index.
        self.nodes_map = {}
        def _walk(node):
            self.nodes_map[node.id] = node
            for child in node.children:
                child.parent = node
                _walk(child)
        _walk(self.root)

    def get_node(self, node_id: str) -> SessionNode | None:
        """Look up a node by id (None if unknown)."""
        return self.nodes_map.get(node_id)

    @property
    def current_node(self) -> SessionNode:
        """The node the cursor points at; falls back to the root if the id is stale."""
        if self.current_id not in self.nodes_map:
            # The index may be stale if nodes were attached directly; rebuild once.
            # Fallback if map out of sync
            self._rebuild_maps()
        return self.nodes_map.get(self.current_id, self.root)

    def add_message(self, role: str, content: str, attachments: list | None = None, thought: str = None) -> SessionNode:
        """Append a message as a child of the current node and move the cursor to it."""
        # New messages always attach below the cursor: sending after moving the cursor to an older node creates a branch.
        parent = self.current_node
        child = parent.add_child(role, content, attachments=attachments, thought=thought)

        # Update the index incrementally instead of rebuilding the whole map for every message.
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
        # Unknown ids return None instead of raising so UI handlers can ignore stale clicks.
        if not self.set_current_node(node_id):
            return None

        # Without a role this is a pure cursor move ("switch to that branch"); with a role it also starts a new branch there.
        if role is None:
            return self.current_node
        return self.add_message(role, content, attachments=attachments, thought=thought)

    def get_active_path(self) -> list[SessionNode]:
        """Return nodes from the first message down to the current node (the dummy root is excluded).
        """
        # Walk child -> root via parent pointers (cheap) and reverse, rather than searching down from the root.
        path = []
        curr = self.current_node
        while curr is not None:
            # The dummy root is not a real message and must never reach the model prompt.
            if curr.id != self.root.id:  # Skip the dummy root
                path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def get_active_path_dicts(self) -> list[dict]:
        """Return the active path as plain role/content/id dicts (with attachments when present).
        """
        items = []
        for n in self.get_active_path():
            # Only role/content/id (and attachments) are exposed: this shape is what inference code and prompt builders consume.
            item = {"role": n.role, "content": n.content, "id": n.id}
            if n.attachments:
                item["attachments"] = n.attachments
            items.append(item)
        return items

    def set_current_node(self, node_id: str) -> bool:
        """Move the cursor to node_id; returns False if the node does not exist."""
        # A node attached behind our back (e.g. tests or from_dict surgery) is missing from the index; one
        # rebuild is cheaper than failing the cursor move.
        if node_id not in self.nodes_map:
            self._rebuild_maps()
        if node_id in self.nodes_map:
            self.current_id = node_id
            return True
        return False

    def update_current_node_content(self, text: str):
        """Replace the text of the current node (used while streaming a reply)."""
        # Called for every streamed token batch, so it deliberately does no validation or index work.
        self.current_node.content = text

    def update_node_content(self, node_id: str, text: str) -> bool:
        """Replace a node's text by id; returns False if the node does not exist."""
        node = self.get_node(node_id)
        if node is None:
            return False
        node.content = text
        return True

    def node_depth(self, node_id: str | None = None) -> int:
        """Number of messages between the root and the node (default: current node)."""
        node = self.get_node(node_id or self.current_id)
        depth = 0
        while node is not None and node.id != self.root.id:
            depth += 1
            node = node.parent
        return depth

    def leaf_nodes(self) -> list[SessionNode]:
        """Return all message nodes with no message children (the tips of every branch)."""
        leaves: list[SessionNode] = []

        # Recursion depth equals conversation depth, which is small for chat sessions, so no explicit stack is needed.
        def _walk(node: SessionNode):
            # Filtering out the root id is defensive: a root never appears as a child in normal trees.
            message_children = [child for child in node.children if child.id != self.root.id]
            if node.id != self.root.id and not message_children:
                leaves.append(node)
            for child in node.children:
                _walk(child)

        _walk(self.root)
        return leaves

    def stats(self) -> SessionTreeStats:
        """Compute total, message, leaf and max-depth counts for the tree."""
        total = 0
        message_nodes = 0
        max_depth = 0

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
            # leaf_count reuses leaf_nodes() so the two definitions of "leaf" cannot drift apart.
            total_nodes=total,
            message_nodes=message_nodes,
            leaf_count=len(self.leaf_nodes()),
            max_depth=max_depth,
        )

    def active_branch_label(self) -> str:
        """Short label like 'assistant:1a2b3c4d' for the active leaf, or 'root' when empty."""
        path = self.get_active_path()
        if not path:
            return "root"

        # First 8 hex characters of the id are enough to tell sibling branches apart in the UI label.
        leaf = path[-1]
        return f"{leaf.role}:{leaf.id[:8]}"

    def to_dict(self) -> dict:
        """Serialize the tree and cursor position."""
        return {
            "root": self.root.to_dict(),
            "current_id": self.current_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SessionTree':
        """Rebuild a tree from serialized data; empty or malformed data yields a fresh tree."""
        # Corrupt or legacy session files must not crash the Workbench; they degrade to an empty conversation.
        if not data or "root" not in data:
            return cls()
        root = SessionNode.from_dict(data["root"])
        return cls(root, data.get("current_id"))

    def clear(self):
        """Reset to a fresh empty tree."""
        self.root = SessionNode(role="system", content="Root", node_id="root")
        self.current_id = self.root.id

        # The index is rebuilt by hand here because clear() replaced the root without calling _rebuild_maps().
        self.nodes_map = {self.root.id: self.root}

    def copy(self) -> 'SessionTree':
        """Deep copy via a serialize/deserialize round trip."""
        # Round-tripping through dicts avoids copy.deepcopy following the parent pointers (parent <-> children
        # cycles) and gives the copy fresh parent links and index.
        return SessionTree.from_dict(self.to_dict())

    # The methods below make the tree usable wherever code expects a list of message dicts (the active branch).
    # List interface duck-typing helpers
    def __len__(self):
        """Number of messages on the active path."""
        return len(self.get_active_path())

    def __getitem__(self, index):
        """Index or slice into the active path's message dicts."""
        active = self.get_active_path_dicts()
        return active[index]

    def __iter__(self):
        """Iterate over the active path's message dicts."""
        return iter(self.get_active_path_dicts())

    def __bool__(self):
        """True when the active path contains at least one message."""
        # Defining __bool__ keeps `if chat_history:` working: an empty conversation is falsy, as an empty list was.
        return len(self.get_active_path()) > 0

    def append(self, msg: dict):
        """Add a message dict (needs 'role' and 'content') after the current node."""
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            # Fail loudly: silently dropping a malformed message would corrupt the conversation.
            raise TypeError("Appended message must be a dict containing 'role' and 'content'")
        self.add_message(msg["role"], msg["content"], attachments=msg.get("attachments"))

    # Default folder for saved sessions (relative to the working directory).
    SESSIONS_DIR = "data/sessions"

    def save(self, session_id: str | None = None) -> str:
        """Write this tree to data/sessions/{session_id}.json. Returns the path."""
        import os
        import json

        # Created on demand so a fresh install can save its first session.
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)
        if session_id is None:
            import uuid

            # Short ids are friendlier in file names and lists; collisions across a few thousand sessions are very unlikely.
            session_id = str(uuid.uuid4())[:8]
        path = os.path.join(self.SESSIONS_DIR, f"{session_id}.json")

        # Serialised first, then stripped, so the in-memory tree is never modified by saving.
        payload = self.to_dict()
        # Reasoning traces are removed from the saved copy only; the in-memory tree keeps them.
        _strip_thoughts(payload["root"])
        payload["session_id"] = session_id
        # The first 80 characters of the first user message become the preview shown in session lists.
        # Store first user message as preview
        for node_data in payload.get("root", {}).get("children", []):
            if node_data.get("role") == "user":
                payload["preview"] = node_data.get("content", "")[:80]
                break
        import os as _os
        # Write-then-rename so an interrupted save cannot corrupt an existing session file.
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

        # Files written before session ids existed load with the placeholder id "unknown".
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

        # Reverse name order is only a cheap first ordering; the authoritative sort by modification time follows below.
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
            # One corrupt file (partial write from a crash, hand-edited JSON) must not hide every other session
            # from the list, so unreadable or non-session JSON files are skipped silently.
            except Exception:
                pass

        # Newest-first by file modification time, which also reflects later edits to an old session.
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return sessions

    def delete_session_file(self, path: str):
        """Delete a session file if it exists."""
        import os

        # Idempotent: deleting an already-removed file (e.g. from two UI actions) is not an error.
        if os.path.exists(path):
            os.remove(path)
