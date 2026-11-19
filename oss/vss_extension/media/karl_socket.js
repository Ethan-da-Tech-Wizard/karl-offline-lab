// ==============================================================================
// Program: karl_socket.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: WebSocket connection manager maintaining real-time duplex RPC communication with Karl backend.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// ── Bridge Relay Layer ─────────────────────────────────────────────────────────
// All outbound RPC passes through BridgeRelay. When window.KARL_USE_HOST_RELAY
// is true the extension host owns the WebSocket and we communicate via
// postMessage. When false (the default), we manage the WebSocket directly.
// The relay flag lets the host opt-in without any webview code changes.

/**
 * Facade over the two ways of talking to the Karl backend.
 *
 * Direct mode: this page owns a WebSocket and sends JSON itself. Host-relay mode
 * (``window.KARL_USE_HOST_RELAY``): the extension host owns the socket and this page forwards
 * frames with postMessage. Callers only use send() and connected() and need not know which.
 */
const BridgeRelay = (() => {
    /**
     * Send ``payload`` as JSON over the page's own WebSocket; silently dropped when it is not open.
     * @param {object} payload
     */
    function _directSend(payload) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(payload));
        }
    }

    return {
        // Relay mode hands the frame to the extension host; otherwise write to the socket directly.
        send(payload) {
            if (window.KARL_USE_HOST_RELAY) {
                vscode.postMessage({ command: 'bridge_send', payload });
            } else {
                _directSend(payload);
            }
        },
        // In relay mode the page cannot see the socket, so it trusts the connection state the host reported.
        connected() {
            if (window.KARL_USE_HOST_RELAY) return connectionState === 'connected';
            return socket !== null && socket.readyState === WebSocket.OPEN;
        }
    };
})();

// ── token refresh state ───────────────────────────────────────────────────────

// Session-token lifetime handling: the backend issues a token valid 12 hours; the page asks for a
// replacement 10 minutes before it expires, so a long-open panel never loses its connection.
let tokenIssuedAt = null;       // epoch-ms when the current session token was issued
let tokenRefreshTimer = null;   // setInterval handle for the proactive refresh check

const _TOKEN_LIFETIME_MS = 12 * 60 * 60 * 1000;    // 12 hours in ms
const _REFRESH_BEFORE_MS = 10 * 60 * 1000;          // trigger 10 min before expiry

/**
 * Start (or restart) the once-a-minute check that renews the session token.
 */
function _startTokenRefreshTimer() {
    _clearTokenRefreshTimer();
    tokenRefreshTimer = setInterval(_checkTokenRefresh, 60_000); // check every minute
}

/**
 * Stop the token refresh check, if running.
 */
function _clearTokenRefreshTimer() {
    if (tokenRefreshTimer) {
        clearInterval(tokenRefreshTimer);
        tokenRefreshTimer = null;
    }
}

/**
 * If the token is within 10 minutes of expiry, send the ``refresh_token`` RPC (id 90).
 *
 * Does nothing when no token was issued yet or the socket is not connected.
 */
function _checkTokenRefresh() {
    if (!tokenIssuedAt || !isConnected()) return;
    const elapsed = Date.now() - tokenIssuedAt;
    if (elapsed >= _TOKEN_LIFETIME_MS - _REFRESH_BEFORE_MS) {
        // Prefer the cached token (possibly rotated by an earlier refresh); empty string if none.
        const currentToken = (boot && boot.token) ? boot.token : '';
        rpc(90, 'refresh_token', { token: currentToken });
    }
}

// ── connection state ──────────────────────────────────────────────────────────

/**
 * Update every status indicator in the UI (dot, labels, offline panel).
 * @param {string} state one of offline, connecting, connected, running or error (used as a CSS class)
 * @param {string} label human-readable text
 */
function setConnectionState(state, label) {
    connectionState = state;
    $('statusDot').className = `status-dot ${state}`;
    $('statusText').innerText = label;
    $('cockpitConnection').innerText = label.toLowerCase();
    $('cockpitConnection').className = `state-chip ${state}`;
    // Show the "start Karl" help panel whenever we are neither connected nor busy running.
    $('offlinePanel').classList.toggle('active', state !== 'connected' && state !== 'running');
    updateBridgeMeta();
}

/**
 * Stop all timers and detach and close the WebSocket, then mark the UI offline.
 * @param {boolean} [isError] show "Error" instead of "Offline"
 */
function teardownSocket(isError = false) {
    _clearTokenRefreshTimer();
    tokenIssuedAt = null;
    if (runtimeStatusTimer) {
        clearInterval(runtimeStatusTimer);
        runtimeStatusTimer = null;
    }
    if (socket) {
        // Detach handlers first so closing does not trigger onclose logic (which would start a reconnect loop).
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        try {
            // Closing an already-dead socket can throw; the empty catch below makes that harmless.
            if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
                socket.close();
            }
        } catch {}
        socket = null;
    }
    setConnectionState('offline', isError ? 'Error' : 'Offline');
    renderRuntimeOffline();
}

/**
 * Clean up after a lost connection and, if allowed, schedule an automatic reconnect in 5 s.
 *
 * No retry happens when the user disconnected on purpose, autoConnect is off, or a retry is
 * already counting down.
 * @param {boolean} [isError]
 */
function handleDisconnect(isError = false) {
    teardownSocket(isError);
    if (boot.autoConnect && !manualDisconnect && !reconnectTimer) {
        nextReconnectSec = 5;
        log(`[Bridge] Connection lost. Retrying in ${nextReconnectSec}s.`);
        updateBridgeMeta();
        // Tick once a second so the UI can show a live countdown; connect() runs when it hits 0.
        reconnectTimer = setInterval(() => {
            nextReconnectSec--;
            updateBridgeMeta();
            if (nextReconnectSec <= 0) {
                clearInterval(reconnectTimer);
                reconnectTimer = null;
                connect();
            }
        }, 1000);
    }
}

// ── public connection API ──────────────────────────────────────────────────────

// connect() is the single entry point. It dispatches to the host relay or to
// the direct WebSocket implementation depending on KARL_USE_HOST_RELAY.
/**
 * Open the connection to the Karl backend (single entry point).
 *
 * In relay mode it asks the extension host to connect; otherwise it opens the WebSocket here.
 */
function connect() {
    if (window.KARL_USE_HOST_RELAY) {
        manualDisconnect = false;
        const port = Number($('bridgePort').value) || boot.port || 8080;
        $('cockpitPort').innerText = port;
        setConnectionState('connecting', 'Connecting');
        log(`[Bridge] Requesting host relay connection to ws://localhost:${port}`);
        vscode.postMessage({ command: 'bridge_connect', port });
        return;
    }
    _directConnect();
}

/**
 * User-initiated disconnect: cancel retry and status timers, then close the connection.
 *
 * Sets ``manualDisconnect`` so handleDisconnect() does not reconnect automatically.
 */
function disconnect() {
    manualDisconnect = true;
    if (reconnectTimer) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
    }
    if (bridgeMetaTimer) {
        clearInterval(bridgeMetaTimer);
        bridgeMetaTimer = null;
    }
    if (window.KARL_USE_HOST_RELAY) {
        vscode.postMessage({ command: 'bridge_disconnect' });
        setConnectionState('offline', 'Offline');
        return;
    }
    teardownSocket(false);
}

// ── direct WebSocket implementation ───────────────────────────────────────────

/**
 * Create the WebSocket to ws://localhost:<port> (token as a query parameter) and wire its handlers.
 *
 * The log line masks the token. The close codes 4002 (lease expired) and 4003 (token revoked)
 * are custom backend codes with different recovery behaviour.
 */
function _directConnect() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        teardownSocket(false);
    }
    manualDisconnect = false;
    const port = Number($('bridgePort').value) || boot.port || 8080;
    $('cockpitPort').innerText = port;
    setConnectionState('connecting', 'Connecting');
    const token = (boot && boot.token) ? boot.token : '';
    // Only the loopback host is used: the bridge is a local service, never a remote one.
    const wsUrl = `ws://localhost:${port}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
    log(`[Bridge] Connecting to ${wsUrl.replace(token, '***')}`);

    try {
        socket = new WebSocket(wsUrl);
    } catch (err) {
        lastBridgeError = err.message;
        handleDisconnect(true);
        return;
    }

    // On connect: reset timers, fetch runtime status/agent profiles/adapters, then poll status every 4 s.
    socket.onopen = () => {
        lastConnectedAt = new Date();
        tokenIssuedAt = Date.now();
        setConnectionState('connected', 'Connected');
        log('[Bridge] Connected.');
        if (reconnectTimer) {
            clearInterval(reconnectTimer);
            reconnectTimer = null;
        }
        requestRuntimeStatus();
        loadAgentProfiles();
        loadAdapters();
        runtimeStatusTimer = runtimeStatusTimer || setInterval(requestRuntimeStatus, 4000);
        _startTokenRefreshTimer();
        persist();
    };

    // One bad frame must not kill the connection, so parse errors are logged and ignored.
    socket.onmessage = event => {
        try {
            const data = JSON.parse(event.data);
            handleSocketMessage(data);
        } catch (err) {
            lastBridgeError = 'Malformed frame: ' + err.message;
            log(`[Bridge Error] Malformed JSON: ${err.message}`);
            updateBridgeMeta();
        }
    };

    socket.onerror = () => {
        lastBridgeError = 'Connection failed';
        vscode.postMessage({ command: 'show_error', text: 'Karl bridge connection failed. Start Karl and verify the WebSocket port.' });
    };

    socket.onclose = (event) => {
        // Lease expired: clearing the cached token makes the reconnect fetch a fresh one.
        if (event.code === 4002) {
            log('[Bridge] Session lease expired (4002). Re-reading token and reconnecting.');
            // Token was rotated server-side; clear our cached copy so the
            // next connect() re-reads bridge_token.json from disk.
            if (boot) boot.token = '';
            handleDisconnect(false);
            return;
        }
        // Revoked: reconnecting automatically would only be rejected again, so stop and wait for the user.
        if (event.code === 4003) {
            log('[Bridge] Token revoked by server (4003). Manual reconnect required.');
            manualDisconnect = true;
            teardownSocket(true);
            return;
        }
        handleDisconnect(false);
    };
}

// ── RPC ───────────────────────────────────────────────────────────────────────

/**
 * Send a JSON-RPC 2.0 request to the backend.
 *
 * The numeric ``id`` is a fixed per-feature code: handleRpcResult() uses it to decide what to do with
 * the reply. Calls made while offline show an error, except ids 20/31/40/50 (the passive data
 * loaders), which stay quiet.
 * @param {number} id
 * @param {string} method
 * @param {object} [params]
 */
function rpc(id, method, params) {
    if (!BridgeRelay.connected()) {
        if (![20, 31, 40, 50].includes(id)) {
            vscode.postMessage({ command: 'show_error', text: 'Karl bridge is offline.' });
        }
        return;
    }
    BridgeRelay.send({ jsonrpc: '2.0', id, method, params });
}

/**
 * True when the bridge connection is open.
 */
function isConnected() {
    return BridgeRelay.connected();
}

// ── incoming message dispatcher ───────────────────────────────────────────────

/**
 * Dispatch one message from the backend: errors, replies to our requests, or pushed notifications.
 *
 * Replies carry ``result`` and go to handleRpcResult(). Notifications carry ``method`` and
 * ``params`` (swarm progress, edit proposals, knowledge-base ingestion, streamed chat tokens...).
 * @param {any} data parsed JSON frame
 */
function handleSocketMessage(data) {
    // Error replies: show them, and re-enable the knowledge-base button if ingestion (id 51) failed.
    if (data.error) {
        const message = data.error.message || 'Unknown Karl bridge error.';
        lastBridgeError = message;
        log(`[Bridge Error] ${message}`);
        vscode.postMessage({ command: 'show_error', text: message });
        if (data.id === 51) {
            $('kbIngestBtn').disabled = false;
            $('kbIngestState').innerText = 'Error';
        }
        return;
    }

    // Anything with a result is a reply to one of our numbered requests.
    if (data.result) {
        handleRpcResult(data.id, data.result);
        return;
    }

    const method = data.method;
    const params = data.params || {};

    if (method === 'status_update') {
        log(params.message || '');
        addTimeline('Status', params.message || '');

    } else if (method === 'task_plan_created') {
        addTimeline('Plan', 'Swarm architect created an implementation plan.');

    } else if (method === 'agent_profiles_updated') {
        swarmAgentProfiles = params.profiles || swarmAgentProfiles || {};
        renderAgentProfileOptions();
        log('[Swarm] Agent profiles updated.');

    } else if (method === 'edits_proposed') {
        // The swarm only proposes edits; each one is queued for the user to approve before it is written.
        const proposals = Array.isArray(params.proposals) ? params.proposals : [];
        // Unique id grouping all proposals from one swarm run.
        const batchId = `swarm-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        addTimeline('Review', `Swarm proposed ${proposals.length} file edit(s).`);
        proposals.forEach(proposal => {
            // Skip malformed proposals instead of failing the whole batch.
            if (!proposal || !proposal.filepath || typeof proposal.content !== 'string') return;
            log(`[Review] Awaiting approval for ${proposal.filepath}`);
            vscode.postMessage({
                command: 'queue_file_edit',
                filepath: proposal.filepath,
                content: proposal.content,
                summary: proposal.summary || 'Swarm proposed a verified file update.',
                swarmBatchId: batchId
            });
        });
        switchWorkspace('changes');

    } else if (method === 'file_edited') {
        log(`[Edit] Karl wrote approved change for ${params.filepath}`);
        addTimeline('Write', `Applied approved edit: ${params.filepath}`);
        if (typeof markPendingEditByPath === 'function') {
            markPendingEditByPath(params.filepath, 'written');
        }

    } else if (method === 'test_result') {
        const status = params.passed ? 'passed' : 'failed';
        addTimeline('Test', `Verification ${status}.`);
        log(`[Test] ${status.toUpperCase()}`);
        if (!params.passed && params.error_trace) log(params.error_trace);

    } else if (method === 'proposal_verification_finished') {
        const status = params.passed ? 'passed' : 'failed';
        addTimeline('Dry Run', `Layer ${params.layer} proposal verification ${status}.`);
        log(`[Dry Run] Layer ${params.layer} ${status.toUpperCase()} before approval.`);
        if (!params.passed && params.trace) log(params.trace);

    } else if (method === 'finished_swarm') {
        setConnectionState('connected', 'Connected');
        // Swarm done: update the task state, then record the run (changed files are parsed from the "Modified files: a, b" summary).
        finishActiveTask(params.success ? 'completed' : 'failed');
        addTimeline('Finished', params.success ? 'Swarm finished successfully.' : 'Swarm finished with issues.');
        log(`[Swarm] ${params.success ? 'SUCCESS' : 'FAILURE'}: ${params.summary || ''}`);
        if (params.run_id) {
            const changedFiles = String(params.summary || '').replace(/^Modified files:\s*/, '').split(',')
                .map(item => item.trim())
                .filter(item => item && item !== 'No files modified successfully.');
            vscode.postMessage({
                command: 'record_swarm_run',
                run: {
                    runId: params.run_id,
                    timestamp: new Date().toISOString(),
                    objective: $('objective') ? $('objective').value.trim() : '',
                    summary: params.summary || '',
                    changedFiles
                }
            });
            log(`[Swarm] Run ID: ${params.run_id}`);
        }

    } else if (method === 'kb_ingest_progress') {
        $('kbIngestState').innerText = `${params.current}/${params.total}`;
        log(`[KB] ${params.current}/${params.total}: ${params.filename}`);

    } else if (method === 'kb_ingest_finished') {
        $('kbIngestBtn').disabled = false;
        $('kbIngestState').innerText = params.error_count ? 'Check Log' : 'Ready';
        markCurrentKbQueueDone(params.error_count ? 'error' : 'done');
        renderKbSnapshot(params.snapshot || {});
        log(`[KB] Added ${params.chunks_added} chunk(s) from ${params.file_count} file(s).`);
        (params.errors || []).forEach(err => log(`[KB] ${err.filename}: ${err.error}`));
        if (kbQueueRunning) ingestNextQueuedPath();

    } else if (method === 'chat_thought_token') {
        // Streaming chat: thought tokens go to the reasoning panel, answer tokens (next branch) to the chat view.
        // Route thought tokens to the collapsible reasoning panel
        if (!labRunning) {
            $('introspectionBox').classList.add('active');
            appendThoughtToken(params.token || '');
        }

    } else if (method === 'chat_response_token') {
        handleResponseToken(params.token || '');

    } else if (method === 'chat_finished') {
        handleChatFinished();

    } else if (method === 'vision_result') {
        $('visionResult').style.display = 'block';
        $('visionResultText').innerText = `Caption: ${params.caption || 'No caption'}\n\nOCR Text:\n${params.ocr || 'No OCR text detected.'}`;
    }
}

/**
 * Handle the reply to a numbered request.
 *
 * Request ids: 90 token refresh; 12 prompt-lab diff; 30 runtime status (also used to measure
 * ping latency); 31 list models; 32 model change; 33 list adapters; 40-43 prompt pairs
 * (list/load/save/delete); 50-53 knowledge base (sources/ingest/search/swarm RAG); 70-71 agent
 * profiles; 20-21 codex topics/content. A rollback result is recognised by its shape.
 * @param {number} id
 * @param {any} result
 */
function handleRpcResult(id, result) {
    if (!result || typeof result !== 'object') result = {};
    // Only the token is stored; nothing is shown to the user.
    if (id === 90) {
        // refresh_token response — update cached token and reset the lease timer
        if (result.token) {
            if (boot) boot.token = result.token;
            tokenIssuedAt = Date.now();
            log('[Bridge] Token refreshed. Next refresh in ~11h50m.');
        }
        return;
    }
    if (id === 12) {
        renderLabDiff(labOutputA, labOutputB);
        labRunning = false;
        $('labRunBtn').disabled = false;
        log('[Prompt Lab] Diff complete.');
    } else if (id === 30) {
        lastHeartbeatAt = new Date();
        // Round-trip time of the last status request, shown in the runtime panel.
        const latency = pingStartTimestamp ? (Date.now() - pingStartTimestamp) : 0;
        renderRuntimeStatus(result, latency);
        updateBridgeMeta(result);
    } else if (id === 31) {
        renderModels(result.models || []);
        renderDownloadRegistry(result.models || []);
    } else if (id === 32) {
        log(`[Models] ${result.message || 'Model updated.'}`);
        requestRuntimeStatus();
        loadModels();
        loadAdapters();
    } else if (id === 33) {
        // Keep the adapter currently shown as active selected when the list is redrawn.
        const activeAdapter = ($('runtimeAdapter') && $('runtimeAdapter').innerText !== 'none') 
            ? $('runtimeAdapter').innerText 
            : '';
        renderAdapters(result.adapters || [], activeAdapter);
    } else if (id === 40) {
        renderPromptPairs(result.pairs || []);
    } else if (id === 41) {
        applyPromptPair(result);
    } else if (id === 42) {
        $('promptPairName').value = result.name || '';
        log(`[Prompt Lab] Saved pair ${result.name || ''}`);
        loadPromptPairs();
    } else if (id === 43) {
        log(`[Prompt Lab] Deleted pair ${result.name || ''}`);
        $('promptPairName').value = '';
        loadPromptPairs();
    } else if (id === 50) {
        renderKbSnapshot(result);
    } else if (id === 51) {
        $('kbIngestBtn').disabled = true;
        $('kbIngestState').innerText = 'Running';
        log(`[KB] Ingestion started for ${result.file_count || 0} file(s).`);
    } else if (id === 52) {
        renderKbSearch(result);
    } else if (id === 53) {
        renderSwarmRagResults(result);
    } else if (id === 70) {
        swarmAgentProfiles = result || {};
        renderAgentProfileOptions();
        log('[Swarm] Agent profiles loaded.');
    } else if (id === 71) {
        log(`[Swarm] Saved agent profile ${result.id || ''}.`);
        loadAgentProfiles();
    } else if (typeof result.ok === 'boolean' && Array.isArray(result.restored) && Array.isArray(result.removed)) {
        const status = result.ok ? 'complete' : 'failed';
        addTimeline('Rollback', `Swarm rollback ${status}.`);
        log(`[Rollback] ${result.message || status}`);
        if (result.restored.length) log(`[Rollback] Restored: ${result.restored.join(', ')}`);
        if (result.removed.length) log(`[Rollback] Removed: ${result.removed.join(', ')}`);
    } else if (id === 20) {
        renderCodexTopics(result.topics || []);
    } else if (id === 21) {
        const cleanText = result.content || '';
        // The text is HTML-escaped before insertion, so reference content cannot inject markup.
        $('codexViewer').innerHTML = `
            <div class="action-row compact-actions" style="margin-bottom:8px;">
                <button id="codexSendChatBtn">Send to Chat</button>
                <button id="codexSendSwarmBtn">Send to Swarm</button>
            </div>
            <div class="codex-content">${escapeHtml(cleanText)}</div>
        `;
        $('codexSendChatBtn').addEventListener('click', () => {
            const txt = $('codexViewer').querySelector('.codex-content').innerText;
            $('chatInput').value = `Codex Reference:\n\n${txt}\n\n`;
            switchWorkspace('chat');
        });
        $('codexSendSwarmBtn').addEventListener('click', () => {
            const txt = $('codexViewer').querySelector('.codex-content').innerText;
            $('objective').value = `Codex Reference:\n\n${txt}\n\n${$('objective').value}`;
            switchWorkspace('swarm');
        });
    }
}

// ── polling ────────────────────────────────────────────────────────────────────

// Timestamp of the last status request, used to compute latency in handleRpcResult().
let pingStartTimestamp = null;

/**
 * Ask the backend for its runtime status (also acts as a heartbeat) and start the latency timer.
 */
function requestRuntimeStatus() {
    pingStartTimestamp = Date.now();
    rpc(30, 'get_runtime_status');
}

/**
 * Refresh the one-line bridge status text (heartbeat age, last connect, version, retry countdown, last error).
 * @param {any} [status] latest runtime status result, if available
 */
function updateBridgeMeta(status) {
    if (!$('bridgeMeta')) return;
    const heartbeat = lastHeartbeatAt ? `${Math.max(0, Math.round((Date.now() - lastHeartbeatAt.getTime()) / 1000))}s ago` : 'never';
    const connected = lastConnectedAt ? lastConnectedAt.toLocaleTimeString() : 'never';
    const version = (status && status.bridge && status.bridge.version) || $('bridgeMeta').dataset.version || 'unknown';

    // While a reconnect countdown runs, the headline label shows it; when offline with no retry, say the app is not running.
    let reconnectStr = '';
    if (reconnectTimer && nextReconnectSec > 0) {
        reconnectStr = ` · Retrying in ${nextReconnectSec}s`;
        const label = `Reconnecting in ${nextReconnectSec}s`;
        $('statusText').innerText = label;
        $('cockpitConnection').innerText = label.toLowerCase();
    } else if (connectionState === 'offline' || connectionState === 'error') {
        const label = 'Karl app not running';
        $('statusText').innerText = label;
        $('cockpitConnection').innerText = label.toLowerCase();
    }
    const error = lastBridgeError ? ` · Last error: ${lastBridgeError}` : '';
    $('bridgeMeta').innerText = `Heartbeat: ${heartbeat} · Last connect: ${connected} · Version: ${version}${reconnectStr}${error}`;
    $('cockpitHeartbeat').innerText = heartbeat;
}

// ── data loaders ───────────────────────────────────────────────────────────────

/**
 * Request the list of models; shows an offline notice instead when disconnected.
 */
function loadModels() {
    if (!isConnected()) {
        $('modelList').innerHTML = '<div class="model-card"><div class="model-meta">Karl bridge is offline.</div></div>';
        return;
    }
    rpc(31, 'list_models');
}

/**
 * Request the list of LoRA adapters, or reset the dropdown to the baseline model when offline.
 */
function loadAdapters() {
    if (!isConnected()) {
        const select = $('adapterSelect');
        if (select) select.innerHTML = '<option value="">None (Baseline Model)</option>';
        return;
    }
    rpc(33, 'list_adapters');
}

/**
 * Request the saved prompt A/B pairs.
 */
function loadPromptPairs() {
    rpc(40, 'list_prompt_pairs');
}

/**
 * Request the knowledge-base source list.
 */
function loadKbSources() {
    if (!isConnected()) {
        $('kbSourceList').innerHTML = '<div class="source-item">Karl bridge is offline.</div>';
        return;
    }
    rpc(50, 'list_kb_sources');
}

/**
 * Request the codex (reference) topic list.
 */
function loadCodexTopics() {
    rpc(20, 'list_codex_topics');
}
