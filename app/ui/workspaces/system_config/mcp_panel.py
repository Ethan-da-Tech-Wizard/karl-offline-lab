# ==============================================================================
# Program: mcp_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Mcp Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# MCP tab of the System workspace: lists configured MCP (Model Context Protocol)
# tool servers from the config store and lets the user add, remove or restart them.

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QLabel, QLineEdit,
    QMessageBox, QInputDialog,
)

from app.engine import config_store
from .common import _section

logger = logging.getLogger("karl.system_config")

# Mixin for the System workspace. All persistence goes through config_store
# (add/remove/get_mcp_config); this class never writes the MCP config file or
# spawns server processes itself. Only the engine's MCPClientManager does that,
# so edits here take effect for running servers only after "Restart MCP Client".
class McpPanelMixin:
    """Mixin that builds the MCP tab and edits the MCP server configuration."""
    def _build_mcp_tab(self) -> QWidget:
        """Build the MCP tab: server list, add form, action buttons and a connected-tools counter."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        panel = QWidget()
        panel.setObjectName("panel")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(8)
        pl.addWidget(_section("MCP TOOL SERVERS"))
    
        desc = QLabel(
            "MCP servers extend Karl with tools: web search, file system, databases, APIs. "
            "Each server is a stdio subprocess spawned on connection."
        )
        desc.setObjectName("lbl-muted")
        desc.setWordWrap(True)
        pl.addWidget(desc)
    
        # Read-only list of configured servers. There is no row selection, which is why
        # "Remove Selected" asks for a name in a dialog instead of using a selection.
        self._mcp_server_list = QTextBrowser()
        # 140 px is about seven lines: typical setups have a handful of servers, and the
        # fixed height keeps the add form from shifting as servers are added.
        self._mcp_server_list.setFixedHeight(140)
        pl.addWidget(self._mcp_server_list)
    
        add_row = QWidget()
        ar = QHBoxLayout(add_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.setSpacing(6)
        # The name is the key in the "mcpServers" mapping (and the identifier used to
        # remove a server), so it must be unique; adding an existing name replaces it
        # in config_store.
        self._mcp_name_input = QLineEdit()
        self._mcp_name_input.setPlaceholderText("Server name (e.g. brave-search)")
        self._mcp_cmd_input  = QLineEdit()
        self._mcp_cmd_input.setPlaceholderText("Command (e.g. npx @modelcontextprotocol/server-brave-search)")
        # Stretch 1:2 gives the command twice the width of the name; commands such as
        # "npx @modelcontextprotocol/..." are far longer than server names.
        ar.addWidget(self._mcp_name_input, 1)
        ar.addWidget(self._mcp_cmd_input, 2)
        pl.addWidget(add_row)
    
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(6)
        add_btn = QPushButton("Add Server")
        add_btn.setObjectName("btn-primary")
        add_btn.clicked.connect(self._add_mcp_server)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setObjectName("btn-ghost")
        remove_btn.clicked.connect(self._remove_mcp_server)
        restart_btn = QPushButton("Restart MCP Client")
        restart_btn.setObjectName("btn-secondary")
        restart_btn.clicked.connect(self._restart_mcp_client)
        br.addWidget(add_btn)
        br.addWidget(remove_btn)
        br.addWidget(restart_btn)
        pl.addWidget(btn_row)
    
        # Dash placeholder until the first _refresh_mcp_display queries the live client.
        self._mcp_tools_lbl = QLabel("Connected tools: —")
        self._mcp_tools_lbl.setObjectName("lbl-muted")
        pl.addWidget(self._mcp_tools_lbl)
        layout.addWidget(panel)
        layout.addStretch()
        self._refresh_mcp_display()
        return w


    def _refresh_mcp_display(self):
        """Redraw the configured server list and the count of connected tools."""
        cfg = config_store.get_mcp_config()
        # The "mcpServers" key and {command, args} entry shape follow the de-facto MCP
        # client config format used by other MCP hosts, so existing snippets paste in.
        servers = cfg.get("mcpServers", {})
        if not servers:
            self._mcp_server_list.setPlainText("No MCP servers configured.")
            return
        # Each entry is shown as "name: command args" in a read-only browser.
        lines = [f"• {name}: {s.get('command', '')} {' '.join(s.get('args', []))}"
                 for name, s in servers.items()]
        self._mcp_server_list.setPlainText("\n".join(lines))
        # The tool count comes from the live client; if it has not started yet, say so instead of failing.
        # Show connected tool count
        try:
            from app.engine.mcp_client import MCPClientManager
            # get_tool_schemas returns the manager cached list and re-queries the servers
            # only when the cache is marked dirty; if that refresh fails it keeps serving the
            # old list, so the count can be briefly stale. KARL_NO_MCP=1 makes it return [],
            # which shows as 0. The broad except below covers the manager failing to start.
            tools = MCPClientManager.get_tool_schemas()
            self._mcp_tools_lbl.setText(f"Connected tools: {len(tools)}")
        except Exception:
            self._mcp_tools_lbl.setText("Connected tools: client not started")


    def _add_mcp_server(self):
        """Validate the name and command fields, store the new server and refresh the list."""
        name = self._mcp_name_input.text().strip()
        cmd_raw = self._mcp_cmd_input.text().strip()
        if not name or not cmd_raw:
            QMessageBox.warning(self, "MCP", "Server name and command are required.")
            return
        # Naive whitespace split: arguments containing spaces are not supported here.
        # If a path contains spaces the command will be split wrongly; such servers
        # must be added by editing the config file.
        parts = cmd_raw.split()
        command, args = parts[0], parts[1:]
        # Saved only; the subprocess is not spawned here. The user restarts the client
        # when ready so a half-typed or wrong command cannot start something at once.
        config_store.add_mcp_server(name, command, args)
        self._mcp_name_input.clear()
        self._mcp_cmd_input.clear()
        self._refresh_mcp_display()
    

    def _remove_mcp_server(self):
        """Ask for a server name and remove it from the saved configuration."""
        # ok is False on cancel; an empty name is ignored rather than treated as remove
        # everything.
        name, ok = QInputDialog.getText(self, "Remove MCP Server", "Server name to remove:")
        if ok and name.strip():
            from app.engine import config_store
            config_store.remove_mcp_server(name.strip())
            self._refresh_mcp_display()
    

    def _restart_mcp_client(self):
        """Recreate the MCP client so config changes take effect, then refresh the display."""
        try:
            from app.engine.mcp_client import MCPClientManager
            # reset then get_instance builds a fresh manager; invalidate_cache drops stale tool schemas.
            # Order matters: reset tears down the old manager (and its stdio subprocesses)
            # before a new one is created, so two copies of each server never run at once.
            MCPClientManager.reset_instance()
            mgr = MCPClientManager.get_instance()
            mgr.start()
            # Tool schemas are cached between calls; without invalidation the model would
            # keep seeing tools from servers that were removed or whose command changed.
            MCPClientManager.invalidate_cache()
            self._refresh_mcp_display()
            QMessageBox.information(self, "MCP", "MCP client restarted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "MCP Error", str(e))
