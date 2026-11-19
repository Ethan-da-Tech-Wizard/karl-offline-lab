# ==============================================================================
# Program: remote_rpc_client.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Optional remote RPC client for distributed inference offloading.
#          Connects to secondary local network nodes when extra compute or VRAM
#          is available.
# ==============================================================================

"""
Client that lets Karl use a remote Karl instance as its "model" over a WebSocket.

RemoteRPCModel imitates the small part of the llama-cpp ``Llama`` interface that
the generation threads use (tokenize and calling with a prompt). Internally it
sends a JSON-RPC ``submit_chat`` request and converts the remote
chat_thought_token / chat_response_token notifications back into the plain
stream of text chunks llama-cpp would produce, so the local parser and UI work
unchanged. If the remote fails, it can fall back to a local model.
"""

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
    """True for loopback or private-network addresses, or the names localhost/127.0.0.1/::1."""
    if not hostname:
        return False
    try:
        # Raises ValueError for DNS names, handled just below.
        return ip_address(hostname).is_private
    except ValueError:
        return hostname in {"localhost", "127.0.0.1", "::1"}


def _ssl_context_for_url(url: str) -> ssl.SSLContext | None:
    """Return the SSL context for a wss:// URL, or None for plain ws://.

    Certificate checking is disabled only for private/loopback hosts (typically
    self-signed certificates on a LAN); public hosts get normal verification.
    """
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
    """Return ``url`` with a ``token`` query parameter added (existing parameters are kept)."""
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
        """``on_fallback(message)`` is called with the error text when remote generation fails.
        ``local_fallback_factory()`` may return a local model to finish the request instead.
        Raises RemoteRPCError if no server URL is configured.
        """
        self.server_url = server_url.strip()
        self.auth_token = auth_token
        self.on_fallback = on_fallback
        self.local_fallback_factory = local_fallback_factory
        self.handshake_timeout = handshake_timeout
        if not self.server_url:
            raise RemoteRPCError("Remote engine mode is enabled but no server URL is configured.")

    def tokenize(self, data: bytes | str, add_bos: bool = False) -> list[int]:
        """Cheap token-count stand-in (about 1 token per 3 characters) returning a list of dummy ids."""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = data
        # Real tokenization would need the remote model; this is only for UI/trim heuristics.
        # Conservative estimate used only for context HUD and trim heuristics in
        # remote mode. It deliberately avoids touching local llama-cpp.
        count = max(1, len(text) // 3) if text else 0
        if add_bos and count:
            count += 1
        return list(range(count))

    def __call__(self, prompt: str, **kwargs) -> Iterator[dict[str, Any]]:
        """Generate from ``prompt``. With stream=True returns an iterator of llama-cpp style chunks.

        Without stream, the whole stream is consumed and one completion dict is returned.
        """
        if not kwargs.get("stream", False):
            text = "".join(
                chunk["choices"][0].get("text", "")
                for chunk in self._stream(prompt, kwargs)
            )
            return {"choices": [{"text": text, "finish_reason": "stop"}]}
        return self._stream(prompt, kwargs)

    def _stream(self, prompt: str, kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
        # Items: chunk dicts, {"error": msg} on failure, or None meaning "finished".
        """Yield completion chunks produced by a background thread that talks to the server.

        The async WebSocket code runs in its own thread/event loop and hands items back
        through a Queue, so this generator can be consumed from ordinary synchronous code.
        """
        queue: Queue[dict[str, Any] | None] = Queue()
        thread = Thread(
            target=lambda: asyncio.run(self._run_stream(prompt, kwargs, queue)),
            daemon=True,
            name="karl-remote-rpc-stream",
        )
        thread.start()

        while True:
            try:
                # Poll with a timeout so a dead worker thread is noticed instead of hanging forever.
                item = queue.get(timeout=0.25)
            except Empty:
                if not thread.is_alive():
                    raise RemoteRPCError("Remote generation ended without a completion signal.")
                continue
            if item is None:
                break
            # Remote failed: notify, then try the local fallback model with the same sampling options, else raise.
            if "error" in item:
                message = item["error"]
                if self.on_fallback:
                    self.on_fallback(message)
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
        """Async worker: connect, send submit_chat, translate notifications into queue items.

        Always ends by putting None (success) or an error dict on the queue.
        """
        try:
            try:
                # Lazy import: the websockets package is only needed in remote mode.
                from websockets.asyncio.client import connect as ws_connect
            except ImportError as exc:
                raise RemoteRPCError(
                    "Remote engine mode requires the 'websockets' package. "
                    "Install requirements.txt before using RPC proxy mode."
                ) from exc

            # The token is sent both in the URL and in the request params.
            url = _with_token_query(self.server_url, self.auth_token)
            ssl_context = _ssl_context_for_url(url)
            websocket_cm = ws_connect(url, ssl=ssl_context, close_timeout=2)
            websocket = await asyncio.wait_for(
                websocket_cm.__aenter__(),
                timeout=self.handshake_timeout,
            )
            try:
                # Millisecond timestamp doubles as the JSON-RPC request id used to recognise the reply.
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

                # The remote sends thought tokens first, then answer tokens; a synthetic </think> is inserted
                # at the switch so the local StreamingThoughtParser sees a normal think block.
                in_thought = True
                started = False
                async for raw in websocket:
                    msg = json.loads(raw)
                    # The direct reply to submit_chat carries no text; only errors matter.
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
                    # End of generation: close an unclosed think block, send a stop chunk, then the end marker.
                    elif method == "chat_finished":
                        if in_thought:
                            queue.put({"choices": [{"text": "</think>", "finish_reason": None}]})
                        queue.put({"choices": [{"text": "", "finish_reason": "stop"}]})
                        queue.put(None)
                        return
                    elif method == "status_update":
                        message = params.get("message", "")
                        if "[Error]" in message:
                            raise RemoteRPCError(message)

                # Closed without a single token: treat as failure so fallback can kick in.
                if not started:
                    raise RemoteRPCError("Remote connection closed before any tokens were received.")
                queue.put(None)
            # Always close the WebSocket cleanly.
            finally:
                await websocket_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("Remote RPC stream failed: %s", exc)
            queue.put({"error": str(exc)})
