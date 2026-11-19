# ==============================================================================
# Program: mcp_client.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Model Context Protocol (MCP) client implementation.
#          Connects Karl to local tool servers and external resource providers
#          over standard JSON-RPC protocols.
# ==============================================================================

"""
MCP Client Manager — Karl Workbench
====================================
Manages the lifecycle of multiple stdio-based Model Context Protocol (MCP) servers
configured in data/mcp_config.json. Exposes synchronous and asynchronous methods
to list tools and invoke them from sync codebases (like QThreads).
"""

import logging
import os
import json
import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Logger "karl.mcp".
logger = logging.getLogger("karl.mcp")

# Timeouts (seconds). They keep a hung or slow MCP server from freezing the caller:
# sync calls wait at most 8s, connecting 5s, MCP initialize handshake 5s.
_SYNC_TIMEOUT_SECONDS = 8.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_INITIALIZE_TIMEOUT_SECONDS = 5.0

# Process-wide tool-list cache so every chat turn does not re-query every server.
# The dirty flag forces a refresh after invalidate_cache().
_cached_tools: list[dict] | None = None
_tools_cache_dirty: bool = True


class MCPClientManager:
    """Singleton that runs MCP servers (stdio child processes) on a private asyncio loop.

    Qt threads are synchronous, so the async MCP SDK runs on a background event-loop
    thread and the public sync methods submit coroutines to it and wait for the result.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: str = "data/mcp_config.json") -> "MCPClientManager":
        """Return the shared manager, creating it and auto-starting configured servers on first use.

        Set KARL_NO_MCP=1 to create the manager without launching any server.
        """
        with cls._lock:
            if cls._instance is None:
                # Created under the class lock so two threads cannot start duplicate server sets.
                cls._instance = cls(config_path)
                if os.environ.get("KARL_NO_MCP") == "1":
                    logger.info("KARL_NO_MCP=1 detected. Skipping MCP server auto-start.")
                    return cls._instance
                try:
                    # Auto-start failures are logged but never prevent the app from using the manager object.
                    cls._instance.start()
                except Exception as e:
                    logger.warning(f"Failed to auto-start MCP Client Manager: {e}")
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Forces recreation of the singleton on the next call."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    @classmethod
    def get_tool_schemas(cls) -> list[dict]:
        """Return cached tool list; refresh if dirty. Safe to call from any thread."""
        global _cached_tools, _tools_cache_dirty
        if os.environ.get("KARL_NO_MCP") == "1":
            return []
        inst = cls.get_instance()
        if _cached_tools is None or _tools_cache_dirty:
            try:
                _cached_tools = inst.list_tools()
                _tools_cache_dirty = False
            except Exception as e:
                logger.warning(f"Failed to refresh tool cache: {e}")
                # On refresh failure keep serving the old list instead of dropping tools.
                _cached_tools = _cached_tools or []
        return _cached_tools or []

    @classmethod
    def invalidate_cache(cls):
        """Mark the tool cache stale so the next get_tool_schemas() re-queries the servers."""
        global _tools_cache_dirty
        _tools_cache_dirty = True

    def __init__(self, config_path: str = "data/mcp_config.json"):
        """Remember the config path and immediately start the background event-loop thread."""
        self.config_path = config_path
        # server name -> {"session", "command", "args", "stop_event"} for connected servers.
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self._start_loop_thread()

    def _start_loop_thread(self):
        """Starts a background daemon thread running an asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        """Thread target: run the asyncio loop until stop() halts it."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # ── Synchronous Entrypoints ───────────────────────────────────────────────

    def _await_future(self, future):
        """Wait up to the sync timeout for a coroutine result; cancel it and re-raise on timeout."""
        try:
            return future.result(timeout=_SYNC_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            future.cancel()
            raise

    def start(self):
        """Synchronously starts all configured MCP servers."""
        if os.environ.get("KARL_NO_MCP") == "1":
            return
        if not self.loop:
            raise RuntimeError("Event loop not initialized.")
        # Hand the coroutine to the loop thread; this call is the sync/async bridge.
        future = asyncio.run_coroutine_threadsafe(self.async_start(), self.loop)
        return self._await_future(future)

    def stop(self):
        """Synchronously stops all running MCP servers and shuts down the loop."""
        if self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.async_stop(), self.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for servers to stop: {e}")
            
            # call_soon_threadsafe is required because stop() is called from a different thread than the loop.
            # Stop the loop and join the thread
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=2.0)
        self.servers.clear()

    def list_tools(self) -> List[Dict[str, Any]]:
        """Synchronously lists all tools across all connected servers."""
        if not self.loop:
            return []
        # Same bridge pattern; errors in the tool itself come back as a normalized dict rather than an exception.
        # Same bridge pattern as start().
        future = asyncio.run_coroutine_threadsafe(self.async_list_tools(), self.loop)
        return self._await_future(future)

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously invokes a tool and returns the response dictionary."""
        if not self.loop:
            return {"is_error": True, "error": "Event loop not running."}
        future = asyncio.run_coroutine_threadsafe(
            self.async_call_tool(server_name, tool_name, arguments), self.loop
        )
        return self._await_future(future)

    # ── Asynchronous Implementation ───────────────────────────────────────────

    async def async_start(self):
        """Loads configuration and initiates server connections."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config: {e}")
            return

        # Config format matches other MCP clients: {"mcpServers": {name: {command, args, env}}}.
        servers_config = config.get("mcpServers", {})
        for name, cfg in servers_config.items():
            command = cfg.get("command")
            args = cfg.get("args", [])
            env = cfg.get("env")

            if not command:
                logger.warning(f"Server '{name}' configuration missing command.")
                continue

            try:
                await self.async_connect_server(name, command, args, env)
            except Exception as e:
                logger.warning(f"Failed to connect to server '{name}': {e}")

    async def async_connect_server(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """Spawns stdio process for a server and performs MCP initialization."""
        # Already connected: nothing to do.
        if name in self.servers:
            return

        # Each server runs in its own task so its async context managers open and close in one task
        # (anyio, used by the SDK, requires that).
        # Start the task context manager thread safely
        task = asyncio.create_task(self._run_server_lifecycle(name, command, args, env))
        
        try:
            await asyncio.wait_for(
                self._wait_for_server_registration(name, task),
                timeout=_CONNECT_TIMEOUT_SECONDS,
            )
        except Exception:
            task.cancel()
            raise

    async def _wait_for_server_registration(self, name: str, task: asyncio.Task):
        """Wait until a lifecycle task registers a server or exits."""
        # Poll every 10 ms until the lifecycle task has registered the session, or surface its failure.
        while name not in self.servers:
            if task.done():
                exc = task.exception()
                if exc:
                    raise exc
                raise RuntimeError("Server lifecycle task exited before registering.")
            await asyncio.sleep(0.01)

    async def _run_server_lifecycle(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]]):
        """Runs the context manager lifetime loop inside a single dedicated task."""
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        try:
            # stdio_client spawns the server process; ClientSession speaks MCP over its pipes.
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(
                        session.initialize(),
                        timeout=_INITIALIZE_TIMEOUT_SECONDS,
                    )
                    
                    # Set up stop signal and record server session
                    stop_event = asyncio.Event()
                    self.servers[name] = {
                        "session": session,
                        "command": command,
                        "args": args,
                        "stop_event": stop_event,
                    }
                    logger.info(f"Successfully connected to server: {name}")
                    
                    # Park here, keeping both context managers open, until async_stop() sets the event.
                    # Yield execution until stop is explicitly requested
                    await stop_event.wait()
        except Exception as e:
            logger.warning(f"Server '{name}' lifecycle terminated: {e}")
        # Always unregister the server when its lifecycle ends, normally or on error.
        finally:
            self.servers.pop(name, None)

    async def async_stop(self):
        """Tears down all active server connections by setting their stop event."""
        for name in list(self.servers.keys()):
            server = self.servers.get(name)
            if server and "stop_event" in server:
                server["stop_event"].set()

        # 20 polls x 50 ms: wait for lifecycle tasks to remove themselves from self.servers.
        # Wait up to 1 second for all sessions to cleanup and pop
        for _ in range(20):
            if not self.servers:
                break
            await asyncio.sleep(0.05)
        logger.info("All MCP servers stopped.")

    async def async_list_tools(self) -> List[Dict[str, Any]]:
        """Collects tools across all active servers."""
        all_tools = []
        for server_name, server in self.servers.items():
            session = server["session"]
            try:
                res = await session.list_tools()
                for tool in res.tools:
                    all_tools.append({
                        "server_name": server_name,
                        "name": tool.name,
                        "description": tool.description or "",
                        # Normalized keys (server_name, name, description, input_schema) are what tool_executor expects.
                        "input_schema": tool.inputSchema or {},
                    })
            except Exception as e:
                logger.warning(f"Error listing tools for server '{server_name}': {e}")
        return all_tools

    async def async_call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Calls a tool asynchronously and returns a normalized response dict."""
        if server_name not in self.servers:
            return {"is_error": True, "error": f"Server '{server_name}' is not connected."}

        session = self.servers[server_name]["session"]
        try:
            res = await session.call_tool(tool_name, arguments)
            
            # Flatten MCP content blocks: text blocks keep their text, others are stringified.
            # Format content list
            formatted_content = []
            if hasattr(res, "content") and res.content:
                for content_block in res.content:
                    if hasattr(content_block, "text"):
                        formatted_content.append({"type": "text", "text": content_block.text})
                    else:
                        formatted_content.append({"type": "other", "data": str(content_block)})

            return {
                "is_error": getattr(res, "isError", False),
                "content": formatted_content,
                "structured_content": getattr(res, "structuredContent", None),
            }
        except Exception as e:
            return {"is_error": True, "error": f"Tool execution failed: {str(e)}"}
