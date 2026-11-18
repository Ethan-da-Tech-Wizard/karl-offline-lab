# Program: karl-offline-lab - test_memory_repository.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests that MemoryManager autosaves sessions through a mock in-memory repository.

"""Tests that MemoryManager autosaves sessions through a mock in-memory repository."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.memory_manager import MemoryManager

class MockInMemoryRepository:
    """Test group or helper class: mock in memory repository."""
    def __init__(self):
        """Initialise this helper object with its starting state."""
        self.store = {}

    def save(self, session_id: str, session_tree: dict) -> None:
        """Test helper: save."""
        self.store[session_id] = session_tree

    def get(self, session_id: str) -> dict | None:
        """Test helper: get."""
        return self.store.get(session_id)

    def list_all(self) -> list[dict]:
        """Test helper: list all."""
        return [
            {"filename": sid, "data": data, "mtime": 123456789.0}
            for sid, data in self.store.items()
            if sid != "autosave_active.json"
        ]

    def delete(self, session_id: str) -> bool:
        """Test helper: delete."""
        if session_id in self.store:
            del self.store[session_id]
            return True
        return False

def test_memory_manager_autosave_with_mock_repository():
    # Session save/load/list moved entirely to SessionTree.save()/load()/
    # list_sessions() (see tests/test_session_tree.py) -- MemoryManager no
    # longer has its own copy of that logic, only the autosave checkpoint
    # and swarm-history helpers below.
    """Scenario: memory manager autosave with mock repository. Expected: the assertions in this test hold."""
    mock_repo = MockInMemoryRepository()
    manager = MemoryManager(sessions_dir="dummy/dir", repository=mock_repo)

    from app.utils.session_tree import SessionTree
    history = SessionTree()
    history.add_message("user", "hello")

    manager.save_autosave_checkpoint(history, "workbench")
    assert "autosave_active.json" in mock_repo.store

    checkpoint = manager.load_autosave_checkpoint()
    assert checkpoint["active_workspace"] == "workbench"

    manager.clear_autosave_checkpoint()
    assert "autosave_active.json" not in mock_repo.store
