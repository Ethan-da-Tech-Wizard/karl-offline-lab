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

class McpPanelMixin:
    """Mixin that builds the MCP tab and edits the MCP server configuration."""
    def _build_mcp_tab(self) -> QWidget:
        """Build the MCP tab: server list, add form, action buttons and a connected-tools counter."""
        # Step 1: Validate input parameters, operational preconditions, and state
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
    
        self._mcp_server_list = QTextBrowser()
        self._mcp_server_list.setFixedHeight(140)
        pl.addWidget(self._mcp_server_list)
    
        add_row = QWidget()
        ar = QHBoxLayout(add_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.setSpacing(6)
        self._mcp_name_input = QLineEdit()
        self._mcp_name_input.setPlaceholderText("Server name (e.g. brave-search)")
        self._mcp_cmd_input  = QLineEdit()
        self._mcp_cmd_input.setPlaceholderText("Command (e.g. npx @modelcontextprotocol/server-brave-search)")
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
    
        self._mcp_tools_lbl = QLabel("Connected tools: —")
        self._mcp_tools_lbl.setObjectName("lbl-muted")
        pl.addWidget(self._mcp_tools_lbl)
        layout.addWidget(panel)
        layout.addStretch()
        self._refresh_mcp_display()
        return w


    def _refresh_mcp_display(self):
        """Redraw the configured server list and the count of connected tools."""
        # Step 1: Validate input parameters, operational preconditions, and state
        cfg = config_store.get_mcp_config()
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
            # Execute guarded operations and establish recovery boundaries
            from app.engine.mcp_client import MCPClientManager
            tools = MCPClientManager.get_tool_schemas()
            self._mcp_tools_lbl.setText(f"Connected tools: {len(tools)}")
        except Exception:
            self._mcp_tools_lbl.setText("Connected tools: client not started")


    def _add_mcp_server(self):
        """Validate the name and command fields, store the new server and refresh the list."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name = self._mcp_name_input.text().strip()
        cmd_raw = self._mcp_cmd_input.text().strip()
        if not name or not cmd_raw:
            QMessageBox.warning(self, "MCP", "Server name and command are required.")
            return
        # Naive whitespace split: arguments containing spaces are not supported here.
        parts = cmd_raw.split()
        command, args = parts[0], parts[1:]
        config_store.add_mcp_server(name, command, args)
        self._mcp_name_input.clear()
        self._mcp_cmd_input.clear()
        self._refresh_mcp_display()
    

    def _remove_mcp_server(self):
        """Ask for a server name and remove it from the saved configuration."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, ok = QInputDialog.getText(self, "Remove MCP Server", "Server name to remove:")
        if ok and name.strip():
            from app.engine import config_store
            config_store.remove_mcp_server(name.strip())
            self._refresh_mcp_display()
    

    def _restart_mcp_client(self):
        """Recreate the MCP client so config changes take effect, then refresh the display."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            from app.engine.mcp_client import MCPClientManager
            # reset then get_instance builds a fresh manager; invalidate_cache drops stale tool schemas.
            MCPClientManager.reset_instance()
            mgr = MCPClientManager.get_instance()
            mgr.start()
            MCPClientManager.invalidate_cache()
            self._refresh_mcp_display()
            QMessageBox.information(self, "MCP", "MCP client restarted successfully.")
        except Exception as e:
            QMessageBox.critical(self, "MCP Error", str(e))
