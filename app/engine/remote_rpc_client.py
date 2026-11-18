# Program: karl-offline-lab - remote_rpc_client.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Presents a remote Karl inference server (reached over a WebSocket JSON-RPC bridge) as a llama-cpp-style callable that streams text, with optional local fallback on failure.
"""Remote RPC model wrapper for streaming generation through a websocket bridge."""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from ipaddress import ip_address
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger("karl.remote_rpc")


class RemoteRPCError(RuntimeError):
    """Raised when the configured remote inference bridge cannot complete."""


def _is_private_host(hostname: str | None) -> bool:
    """Return True when the host is a private-network address or a local host name."""
    if not hostname:
        return False
    try:
        return ip_address(hostname).is_private
    except ValueError:
        return hostname in {"localhost", "127.0.0.1", "::1"}


def _ssl_context_for_url(url: str) -> ssl.SSLContext | None:
    """Return None for non-secure URLs; for secure WebSocket URLs return a TLS context (certificate checks are relaxed only for private hosts)."""
    parsed = urlparse(url)
    if parsed.scheme != "wss":
        return None
    if _is_private_host(parsed.hostname):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return ssl.create_default_context()


def _with_token_query(url: str, token: str) -> str:
    """Return the URL with the auth token added to (or replacing) its query string."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if token:
        query["token"] = token
    return urlunparse(parsed._replace(query=urlencode(query)))


class RemoteRPCModel:
    """
    llama-cpp compatible enough for Karl's generation thread.

    The remote Karl bridge currently streams thought and answer tokens as
    JSON-RPC notifications. This wrapper converts those channel-specific
    notifications back into a normal text stream so the existing parser,
    logger, and UI signal flow keep working.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str,
        on_fallback=None,
        local_fallback_factory=None,
        handshake_timeout: float = 5.0,
    ):
        """Store the server URL, token, fallback hooks and handshake timeout; fail if no URL is configured."""
        self.server_url = server_url.strip()
        self.auth_token = auth_token
        self.on_fallback = on_fallback
        self.local_fallback_factory = local_fallback_factory
        self.handshake_timeout = handshake_timeout
        if not self.server_url:
            raise RemoteRPCError("Remote engine mode is enabled but no server URL is configured.")

    def tokenize(self, data: bytes | str, add_bos: bool = False) -> list[int]:
        """Return a rough placeholder token list (about one per three characters) used only for size estimates in remote mode."""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = data
        # Conservative estimate used only for context HUD and trim heuristics in
        # remote mode. It deliberately avoids touching local llama-cpp.
        # Rough estimate: about three characters per token.
        count = max(1, len(text) // 3) if text else 0
        if add_bos and count:
            count += 1
        return list(range(count))

    def __call__(self, prompt: str, **kwargs) -> Iterator[dict[str, Any]]:
        """Generate for a prompt: return the stream when stream is requested, otherwise join it into one completion dict."""
        if not kwargs.get("stream", False):
            text = "".join(
                chunk["choices"][0].get("text", "")
                for chunk in self._stream(prompt, kwargs)
            )
            return {"choices": [{"text": text, "finish_reason": "stop"}]}
        return self._stream(prompt, kwargs)

    def _stream(self, prompt: str, kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Run the websocket request on a worker thread and yield its chunks; on an error call the fallback hook and local model if provided, else raise."""
        queue: Queue[dict[str, Any] | None] = Queue()
        thread = Thread(
            target=lambda: asyncio.run(self._run_stream(prompt, kwargs, queue)),
            daemon=True,
            name="karl-remote-rpc-stream",
        )
        thread.start()

        while True:
            try:
                item = queue.get(timeout=0.25)
            except Empty:
                if not thread.is_alive():
                    raise RemoteRPCError("Remote generation ended without a completion signal.")
                continue
            if item is None:
                break
            if "error" in item:
                message = item["error"]
                if self.on_fallback:
                    self.on_fallback(message)
                # On remote failure, optionally answer from a local model instead.
                if self.local_fallback_factory:
                    local_llm = self.local_fallback_factory()
                    local_stream = local_llm(
                        prompt,
                        max_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.95),
                        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                        stream=kwargs.get("stream", True),
                        stop=kwargs.get("stop"),
                        echo=kwargs.get("echo", False),
                    )
                    if isinstance(local_stream, dict):
                        yield local_stream
                        return
                    yield from local_stream
                    return
                raise RemoteRPCError(message)
            yield item

    async def _run_stream(
        self,
        prompt: str,
        kwargs: dict[str, Any],
        queue: Queue[dict[str, Any] | None],
    ) -> None:
        """Connect, send the chat request and translate thought/response notifications into completion-style chunks on the queue."""
        try:
            try:
                from websockets.asyncio.client import connect as ws_connect
            except ImportError as exc:
                raise RemoteRPCError(
                    "Remote engine mode requires the 'websockets' package. "
                    "Install requirements.txt before using RPC proxy mode."
                ) from exc

            url = _with_token_query(self.server_url, self.auth_token)
            ssl_context = _ssl_context_for_url(url)
            websocket_cm = ws_connect(url, ssl=ssl_context, close_timeout=2)
            websocket = await asyncio.wait_for(
                websocket_cm.__aenter__(),
                timeout=self.handshake_timeout,
            )
            try:
                req_id = int(time.time() * 1000)
                params = {
                    "message": prompt,
                    "hyperparams": {
                        "temperature": float(kwargs.get("temperature", 0.7)),
                        "top_p": float(kwargs.get("top_p", 0.95)),
                        "max_tokens": int(kwargs.get("max_tokens", 2048)),
                        "rag_enabled": False,
                        "agentic_loop_enabled": False,
                    },
                    "token": self.auth_token,
                    "rpc_proxy": True,
                }
                await websocket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "submit_chat",
                    "params": params,
                }))

                # The remote stream begins with thought tokens; a closing think tag is injected when the answer starts.
                in_thought = True
                started = False
                async for raw in websocket:
                    msg = json.loads(raw)
                    if msg.get("id") == req_id:
                        if "error" in msg:
                            err = msg["error"].get("message", "Remote submit_chat failed.")
                            raise RemoteRPCError(err)
                        continue

                    method = msg.get("method")
                    params = msg.get("params") or {}
                    text = params.get("token") or ""
                    if method == "chat_thought_token" and text:
                        started = True
                        queue.put({"choices": [{"text": text, "finish_reason": None}]})
                    elif method == "chat_response_token" and text:
                        started = True
                        if in_thought:
                            queue.put({"choices": [{"text": "</think>", "finish_reason": None}]})
                            in_thought = False
                        queue.put({"choices": [{"text": text, "finish_reason": None}]})
                    elif method == "chat_finished":
                        if in_thought:
                            queue.put({"choices": [{"text": "</think>", "finish_reason": None}]})
                        queue.put({"choices": [{"text": "", "finish_reason": "stop"}]})
                        # None tells the consuming generator the stream is complete.
                        queue.put(None)
                        return
                    elif method == "status_update":
                        message = params.get("message", "")
                        if "[Error]" in message:
                            raise RemoteRPCError(message)

                if not started:
                    raise RemoteRPCError("Remote connection closed before any tokens were received.")
                queue.put(None)
            finally:
                await websocket_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("Remote RPC stream failed: %s", exc)
            queue.put({"error": str(exc)})
