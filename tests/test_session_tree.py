# ==============================================================================
# Program: test_session_tree.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising session tree functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# What is under test: app/utils/session_tree.py (SessionNode / SessionTree), the branching conversation model used by the
# Workbench chat, MemoryManager persistence and the branch navigator. Pure data-structure tests: no Qt, no model, no disk
# (the save/load tests at the bottom use pytest's tmp_path/monkeypatch). Run: python -m pytest tests/test_session_tree.py -q
import os
import sys

# Lets the file also be run directly from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.session_tree import SessionNode, SessionTree


# ---------------------------------------------------------------------------
# SessionNode tests
# ---------------------------------------------------------------------------

def test_node_defaults():
    """A new SessionNode has the given role/content, a generated id, no children, no parent and no thought."""
    # Defaults matter because several UI paths create nodes with only role and content.
    node = SessionNode("user", "hello")
    assert node.role == "user"
    assert node.content == "hello"
    assert node.id is not None
    assert node.children == []
    assert node.parent is None
    assert node.thought is None


def test_node_add_child():
    """add_child links both ways (parent.children and child.parent) and keeps the child's role."""
    parent = SessionNode("user", "hi")

    # Both directions are checked: the parent pointer is what get_active_path() walks, the children list is what serialisation walks.
    child = parent.add_child("assistant", "hey")
    assert child in parent.children
    assert child.parent is parent
    assert child.role == "assistant"


def test_node_to_from_dict_round_trip():
    """A node and its children survive to_dict/from_dict including id, content and thought."""
    parent = SessionNode("user", "question", node_id="u1")
    child = parent.add_child("assistant", "answer", node_id="a1")

    # The thought (reasoning trace) is stored separately from content so it can be hidden or stripped on save.
    child.thought = "thinking..."

    data = parent.to_dict()

    # Round trip: if any field were dropped here, a reloaded session would silently lose data.
    restored = SessionNode.from_dict(data)

    assert restored.id == "u1"
    assert restored.content == "question"
    assert len(restored.children) == 1
    assert restored.children[0].id == "a1"
    assert restored.children[0].thought == "thinking..."


def test_node_attachments_round_trip():
    """Attachments (e.g. image id/type) survive serialisation."""
    node = SessionNode(
        "user",
        "what is wrong here?",
        node_id="img1",
        attachments=[{"type": "image", "id": "abc", "path": "data/images/inbox/abc.png"}],
    )
    restored = SessionNode.from_dict(node.to_dict())
    assert restored.attachments[0]["type"] == "image"
    assert restored.attachments[0]["id"] == "abc"


# ---------------------------------------------------------------------------
# SessionTree tests
# ---------------------------------------------------------------------------

def test_tree_empty_on_init():
    """A new SessionTree is empty and falsy (len 0)."""
    tree = SessionTree()
    assert len(tree) == 0

    # Older code writes `if chat_history:`; the tree must stay falsy until a message exists.
    assert not tree  # __bool__


def test_tree_add_message():
    """add_message appends under the current node, so two messages give an active path of user then assistant."""
    tree = SessionTree()
    tree.add_message("user", "Hello")

    # Each add_message moves the cursor, so the second call attaches below the first.
    tree.add_message("assistant", "Hi there")
    path = tree.get_active_path()
    assert len(path) == 2
    assert path[0].role == "user"
    assert path[1].role == "assistant"


def test_tree_active_path_content():
    """The active path exposes dict messages with role, content, id and attachments, the shape the chat view and model prompt use."""
    tree = SessionTree()
    tree.add_message("user", "Q1", attachments=[{"type": "image", "id": "abc"}])
    tree.add_message("assistant", "A1")
    dicts = tree.get_active_path_dicts()
    assert dicts[0]["content"] == "Q1"
    assert dicts[0]["attachments"][0]["id"] == "abc"
    assert dicts[1]["content"] == "A1"

    # The id lets the UI map a rendered bubble back to its node when the user clicks "branch from here".
    assert "id" in dicts[0]


def test_tree_branching():
    """Branch from a node and verify two separate paths exist."""
    tree = SessionTree()
    tree.add_message("user", "Root question")
    root_user_id = tree.current_id

    # The scenario: ask one question, get answer A, go back to the question and get answer B. Both answers must coexist.
    # Path A
    tree.add_message("assistant", "Answer A")
    path_a_id = tree.current_id

    # Branch back to root user message
    # Moving the cursor to an older node is how the UI "rewinds"; the next add_message then forks the conversation.
    tree.set_current_node(root_user_id)
    tree.add_message("assistant", "Answer B")
    path_b_id = tree.current_id

    # Navigate to path A
    tree.set_current_node(path_a_id)
    path_a = tree.get_active_path()
    assert path_a[-1].content == "Answer A"

    # Navigate to path B
    tree.set_current_node(path_b_id)
    path_b = tree.get_active_path()
    assert path_b[-1].content == "Answer B"

    # Both paths share root user message
    # Branches share their common prefix: the same node object, not a copy.
    assert path_a[0].id == path_b[0].id


def test_tree_serialization_round_trip():
    """A whole tree saved to a dict and restored keeps the current node and the active path."""
    tree = SessionTree()
    tree.add_message("user", "ping")
    tree.add_message("assistant", "pong")

    # The cursor position is part of the persisted state; a reloaded session must reopen on the same branch.
    saved_id = tree.current_id

    data = tree.to_dict()
    restored = SessionTree.from_dict(data)

    assert restored.current_id == saved_id
    path = restored.get_active_path()
    assert len(path) == 2
    assert path[0].content == "ping"
    assert path[1].content == "pong"


def test_tree_serialization_with_branch():
    """Serialisation also preserves branches and which branch is active."""
    tree = SessionTree()
    tree.add_message("user", "base")
    base_id = tree.current_id

    tree.add_message("assistant", "branch A")

    tree.set_current_node(base_id)
    tree.add_message("assistant", "branch B")
    b_id = tree.current_id

    data = tree.to_dict()
    restored = SessionTree.from_dict(data)

    # Ids survive the round trip, so a cursor id taken before saving still resolves after loading.
    restored.set_current_node(b_id)
    path = restored.get_active_path()
    assert path[-1].content == "branch B"


def test_tree_clear():
    """clear() empties the tree."""
    tree = SessionTree()
    tree.add_message("user", "hi")
    tree.add_message("assistant", "hello")
    tree.clear()
    assert len(tree) == 0


def test_tree_copy_is_independent():
    """copy() returns an independent tree: adding to the copy leaves the original unchanged."""
    tree = SessionTree()
    tree.add_message("user", "original")

    # copy() round-trips through dicts, so mutating the copy must not touch the original's nodes or index.
    copy = tree.copy()
    copy.add_message("assistant", "added in copy")
    # Original should be unaffected
    assert len(tree) == 1
    assert len(copy) == 2


def test_tree_duck_typing_len_iter():
    """The tree behaves like a list of message dicts for len() and iteration so older list-based code keeps working."""
    tree = SessionTree()
    tree.add_message("user", "a")
    tree.add_message("assistant", "b")
    tree.add_message("user", "c")

    assert len(tree) == 3

    # Iteration yields message dicts, not SessionNode objects, which is what prompt builders expect.
    items = list(tree)
    assert items[0]["role"] == "user"
    assert items[2]["content"] == "c"


def test_tree_getitem():
    """Indexing (including negative) returns active-path messages like a list."""
    tree = SessionTree()
    tree.add_message("user", "first")
    tree.add_message("assistant", "second")
    assert tree[0]["content"] == "first"
    assert tree[1]["content"] == "second"

    # Negative indices work because __getitem__ delegates to a real list.
    assert tree[-1]["content"] == "second"


def test_tree_append_dict():
    """append(dict) is supported for list-style callers."""
    tree = SessionTree()
    tree.append({"role": "user", "content": "via append"})
    tree.append({"role": "assistant", "content": "reply"})
    assert len(tree) == 2
    assert tree[0]["content"] == "via append"


def test_tree_append_invalid_raises():
    """append() with something that is not a message dict raises TypeError."""
    tree = SessionTree()
    try:
        # Contract: append() needs both keys; failing fast avoids a half-built node reaching the prompt.
        tree.append({"role": "user"})  # missing "content"
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_tree_update_current_node_content():
    """The current node's content can be updated in place, used when streaming replaces a placeholder with the final answer."""
    tree = SessionTree()
    tree.add_message("assistant", "draft")

    # Streaming code appends tokens to a placeholder bubble and replaces its text; no new node must be created.
    tree.update_current_node_content("final answer")
    assert tree.current_node.content == "final answer"


def test_tree_nodes_map_consistency():
    """nodes_map must contain every node after tree operations."""
    tree = SessionTree()
    tree.add_message("user", "x")
    uid = tree.current_id
    tree.add_message("assistant", "y")
    tree.set_current_node(uid)
    tree.add_message("assistant", "z")

    # Every node should be in the map
    # Independent tree walk: compares against nodes_map without using the code under test to find nodes.
    def collect(node, ids):
        """Recursively gather every node id below a node."""
        ids.add(node.id)
        for c in node.children:
            collect(c, ids)

    all_ids = set()
    collect(tree.root, all_ids)
    for nid in all_ids:
        assert nid in tree.nodes_map, f"Node {nid} missing from nodes_map"


def test_tree_stats_and_branch_label():
    """Stats report message count, leaf count and depth for a branched tree, and the active branch label starts with the leaf's role."""
    tree = SessionTree()
    user = tree.add_message("user", "base")
    tree.add_message("assistant", "branch A")
    tree.branch_from(user.id, "assistant", "branch B")

    # Expected numbers: 3 messages (base + two answers), 2 leaves (the two answers), depth 2 (base -> answer).
    stats = tree.stats()
    assert stats.message_nodes == 3
    assert stats.leaf_count == 2
    assert stats.max_depth == 2
    assert tree.active_branch_label().startswith("assistant:")


def test_branch_from_missing_returns_none():
    """Branching from an unknown node id returns None and leaves the tree untouched."""
    tree = SessionTree()

    # Stale ids (for example from a deleted branch) must be ignored without side effects.
    assert tree.branch_from("missing", "user", "nope") is None
    assert len(tree) == 0


# ---------------------------------------------------------------------------
# save() / load() / list_sessions() -- the real disk-persistence path
# WorkbenchWorkspace actually uses (MemoryManager.save_session/load_session/
# list_sessions are dead code, superseded by these; see AGENTS.md).
# ---------------------------------------------------------------------------

def test_tree_save_strips_think_blocks(tmp_path, monkeypatch):
    """Saving to disk removes <think> reasoning from stored messages, so private chain-of-thought is not persisted in session files."""
    # Redirect the class-level sessions folder into pytest's temp dir so the tests never touch the real data/sessions.
    # monkeypatch restores the original value after each test, keeping tests independent.
    monkeypatch.setattr(SessionTree, "SESSIONS_DIR", str(tmp_path))

    tree = SessionTree()
    tree.add_message("user", "hello")
    node = tree.add_message(
        "assistant",
        # Both places a trace can hide are populated: inline <think> tags in the content and the separate thought field.
        "<think>\nreasoning about the greeting...\n</think>\nHi there!",
    )
    node.thought = "reasoning about the greeting..."

    path = tree.save("test_session")

    # The returned path is what callers store, so it must be exactly folder/<session_id>.json.
    assert path == os.path.join(str(tmp_path), "test_session.json")

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Checked against the raw file text, not the parsed tree: privacy is about what lands on disk.
    assert "reasoning about the greeting" not in raw
    assert "<think>" not in raw


def test_tree_save_load_round_trip(tmp_path, monkeypatch):
    """save() then load() restores the active path (roles and content) with thoughts dropped."""
    monkeypatch.setattr(SessionTree, "SESSIONS_DIR", str(tmp_path))

    tree = SessionTree()
    tree.add_message("user", "hello")
    tree.add_message("assistant", "<think>\nthinking\n</think>\nHi there!")

    path = tree.save("roundtrip")

    # load() returns the id too, because the file name alone is not authoritative once files are moved or renamed.
    loaded, session_id = SessionTree.load(path)

    assert session_id == "roundtrip"
    active_path = loaded.get_active_path()
    assert [n.role for n in active_path] == ["user", "assistant"]

    # Only the visible answer survives; the think block was stripped when saving.
    assert active_path[1].content == "Hi there!"
    assert active_path[1].thought is None


def test_tree_list_sessions(tmp_path, monkeypatch):
    """list_sessions() reports saved sessions with their id and a preview of the first user message."""
    monkeypatch.setattr(SessionTree, "SESSIONS_DIR", str(tmp_path))

    tree = SessionTree()
    tree.add_message("user", "first question here")
    tree.save("listed_session")

    # list_sessions() reads only the small JSON header fields, which is what the sessions sidebar needs.
    sessions = SessionTree.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "listed_session"
    assert sessions[0]["preview"] == "first question here"


# Lets the file run without pytest (python tests/test_session_tree.py), useful on machines without the dev dependencies.
if __name__ == "__main__":
    test_node_defaults()
    test_node_add_child()
    test_node_to_from_dict_round_trip()
    test_node_attachments_round_trip()
    test_tree_empty_on_init()
    test_tree_add_message()
    test_tree_active_path_content()
    test_tree_branching()
    test_tree_serialization_round_trip()
    test_tree_serialization_with_branch()
    test_tree_clear()
    test_tree_copy_is_independent()
    test_tree_duck_typing_len_iter()
    test_tree_getitem()
    test_tree_append_dict()
    test_tree_append_invalid_raises()
    test_tree_update_current_node_content()
    test_tree_nodes_map_consistency()
    print("All session_tree unit tests PASSED!")
