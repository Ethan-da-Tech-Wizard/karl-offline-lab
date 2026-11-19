"""Regression: get_mcp_config must not leak the module-level default."""
from app.engine import config_store


def test_missing_file_returns_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "missing.json"))
    cfg = config_store.get_mcp_config()
    # Mutating the returned config is the trigger of the bug: if get_mcp_config handed out the module-level default object itself, this write would corrupt the default for everyone.
    cfg["mcpServers"]["x"] = {"command": "y"}
    # The default must remain pristine after the caller mutated its copy.
    assert config_store.MCP_CONFIG_DEFAULT == {"mcpServers": {}}
    assert config_store.get_mcp_config() == {"mcpServers": {}}


def test_add_mcp_server_leaves_default_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "m.json"))
    # add_mcp_server also starts from the default when no file exists, a second path where the shared dict could be mutated in place.
    assert config_store.add_mcp_server("a", "cmd", ["1"])
    # Pointing at a different, missing file makes the next read fall back to the default again; it must not contain server 'a' left over from the previous write.
    monkeypatch.setattr(config_store, "MCP_CONFIG_PATH", str(tmp_path / "other.json"))
    assert config_store.MCP_CONFIG_DEFAULT == {"mcpServers": {}}
    assert config_store.get_mcp_config() == {"mcpServers": {}}
