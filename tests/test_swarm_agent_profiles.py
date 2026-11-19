# ==============================================================================
# Program: test_swarm_agent_profiles.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising swarm agent profiles functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import json

import pytest

from app.engine.swarm_agents import get_tool_schema_block
from app.utils import swarm_agent_profiles as profiles


def test_load_agent_profiles_includes_normalized_defaults(tmp_path, monkeypatch):
    """Loading profiles always includes the built-in architect, coder and tester with normalised fields (builtin flag, tool permissions)."""
    # Profile storage uses the relative path data/agent_profiles.json, so changing the working directory is what isolates each test from the real data folder.
    monkeypatch.chdir(tmp_path)

    loaded = profiles.load_agent_profiles()

    # These three are the roles the swarm pipeline depends on; if a user's file is missing or empty they must still resolve.
    assert set(["architect", "coder", "tester"]).issubset(loaded)
    assert loaded["architect"]["builtin"] is True
    assert loaded["coder"]["tools"]["write_files"] is True
    assert loaded["tester"]["tools"]["execute_sandbox"] is True


def test_save_agent_profile_round_trips_and_clamps(tmp_path, monkeypatch):
    """Saving a custom profile persists it, clamps out-of-range values (temperature capped at 2.0, context_limit floored at 256) and normalises tool permissions."""
    monkeypatch.chdir(tmp_path)

    saved = profiles.save_agent_profile(
        "security_reviewer",
        {
            "name": "Security Reviewer",
            "icon": "SR",
            "system_prompt": "Audit shell and file changes.",
            # Deliberately out-of-range inputs: 9.0 exceeds the 2.0 ceiling and 64 is below the 256-token floor, so the assertions below prove clamping rather than pass-through.
            "temperature": 9.0,
            "context_limit": 64,
            "tools": {"read_files": True, "write_files": False},
        },
    )

    assert saved["temperature"] == 2.0
    assert saved["context_limit"] == 256
    assert saved["tools"]["read_files"] is True
    # A tool the caller did not mention must default to disabled (least privilege), not stay at the built-in default.
    assert saved["tools"]["execute_sandbox"] is False

    data = json.loads((tmp_path / "data" / "agent_profiles.json").read_text(encoding="utf-8"))
    assert "security_reviewer" in data
    assert profiles.load_agent_profiles()["security_reviewer"]["name"] == "Security Reviewer"


def test_save_agent_profile_rejects_invalid_id(tmp_path, monkeypatch):
    """A profile id with illegal characters raises ValueError."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        # The id is used as a dictionary key and could end up in file paths, so path-traversal style ids must be refused up front.
        profiles.save_agent_profile("../bad", {"name": "Bad"})


def test_active_profile_map_resolves_role_overrides(tmp_path, monkeypatch):
    """The active profile map lets a custom profile override a role (coder becomes 'Strict Coder') while other roles keep defaults."""
    monkeypatch.chdir(tmp_path)
    profiles.save_agent_profile(
        "strict_coder",
        {
            "name": "Strict Coder",
            "system_prompt": "Only produce minimal edits.",
            "tools": {"read_files": True, "write_files": False},
        },
    )

    # The map takes role -> profile id overrides; roles absent from it must fall back to their built-ins.
    active = profiles.active_profile_map({"coder": "strict_coder"})

    assert active["coder"]["name"] == "Strict Coder"
    assert active["architect"]["name"] == "Architect"
    assert active["tester"]["name"] == "Tester"


def test_tool_schema_block_respects_allowed_tool_filter():
    """The tool schema shown to a model lists only the tools its profile allows; write and shell tools are omitted for a read-only profile."""
    # The allowed-tool set is the profile's permission list; the prompt must not even advertise tools the agent is forbidden to use, since the model would try to call them.
    schema = get_tool_schema_block({"read_file"})

    assert "name='read_file'" in schema
    assert "name='write_file'" not in schema
    assert "name='shell_run'" not in schema
