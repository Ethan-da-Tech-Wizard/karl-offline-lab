"""Regression: get_mcp_config must not leak the module-level default."""
from app.engine import config_store


def test_missing_file_returns_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "missing.json"))
    cfg = config_store.get_mcp_config()
    cfg["mcpServers"]["x"] = {"command": "y"}
    assert config_store.MCP_CONFIG_DEFAULT == {"mcpServers": {}}
    assert config_store.get_mcp_config() == {"mcpServers": {}}


def test_add_mcp_server_leaves_default_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "m.json"))
    assert config_store.add_mcp_server("a", "cmd", ["1"])
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "other.json"))
    assert config_store.MCP_CONFIG_DEFAULT == {"mcpServers": {}}
    assert config_store.get_mcp_config() == {"mcpServers": {}}
