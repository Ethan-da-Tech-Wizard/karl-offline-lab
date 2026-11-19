// ==============================================================================
// Program: karl.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Client-side controller managing webview chat rendering, prompt submission, and diff previews.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// Where this file sits: it is the LAST script loaded by the webview page built in src/sidebarProvider.js
// (after themes.js, karl_state.js, karl_render.js and karl_socket.js). Those earlier scripts declare the shared
// globals used here without imports: $ (getElementById), vscode (acquireVsCodeApi), boot (window.KARL_BOOTSTRAP),
// rpc(id, method, params), isConnected(), connect()/disconnect(), switchWorkspace(), hydrate()/persist() and the
// render*/update* functions. karl.js itself only wires DOM events and routes messages; it holds no long-lived data.
// Request ids: replies are matched by the numeric id passed to rpc(). Ids used in this file: 1 submit_task,
// 2 stop_task, 21 get_codex_content, 30 heartbeat (host side), 32 set_active_model, 70 get_agent_profiles.
// Keeping ids fixed per method lets karl_socket.js dispatch a reply without storing per-request callbacks.
// Page start-up. Runs once the HTML is parsed. Order matters: appearance and saved state first,
// then event handlers, then the initial (empty) renders; finally tell the extension host the page is
// ready and auto-connect to the backend if enabled.
// All functions called here live in karl_state.js, karl_render.js, karl_socket.js or later in this file.
window.addEventListener('DOMContentLoaded', () => {
    initializeAppearance();
    hydrate();
    _initThoughtsPanel();
    _initChatInnerTabs();
    _initLogsSubtabs();
    _initWsStateObserver();
    bindEvents();
    renderQuickActions();
    renderRecentTasks();
    renderRecentKbQueries();
    renderTaskQueue();
    renderKbQueue();
    renderBranches();
    renderDownloadRegistry([]);
    // Refresh the heartbeat age text every second.
    // Timer handle declared in karl_state.js; kept in a global so the unload handler below can stop it.
    bridgeMetaTimer = setInterval(() => updateBridgeMeta(), 1000);

    // Seeds the chat with a hint so a new user sees what to do before the first connection.
    appendMessageBubble('assistant', 'Karl command cockpit ready. Connect to the local backend to execute agentic workflows.');

    // Handshake: the host can now safely send state to this webview.
    // The host queues messages until this arrives (see KarlSidebarProvider.resolveWebviewView), so it is sent last,
    // after every listener above is installed.
    vscode.postMessage({ command: 'ready' });

    // boot comes from the "karl.autoConnect" setting; connect() opens the page's own WebSocket (karl_socket.js).
    if (boot.autoConnect) connect();
});

// Clean-up when the webview is destroyed: stop timers and close the connection so nothing keeps
// running in the background.
window.addEventListener('unload', () => {
    // Each timer is nulled after clearing so a repeated cleanup is harmless.
    if (bridgeMetaTimer) { clearInterval(bridgeMetaTimer); bridgeMetaTimer = null; }
    if (reconnectTimer)  { clearInterval(reconnectTimer);  reconnectTimer = null; }
    // runtimeStatusTimer and socket are cleaned up by teardownSocket/disconnect
    // Two transport modes exist: the page owns a direct socket (default) or the extension host relays frames
    // (KARL_USE_HOST_RELAY). Shutdown differs accordingly.
    if (window.KARL_USE_HOST_RELAY) {
        vscode.postMessage({ command: 'bridge_disconnect' });
    } else {
        teardownSocket(false);
    }
});

// Single inbox for messages from the extension host (see sidebarProvider.js). Every message has a
// ``command``; most are handled by calling a function and returning early.
window.addEventListener('message', event => {
    // `|| {}` protects against messages with no data (for example from other extensions posting to the window).
    const message = event.data || {};

    // The host rotated the bridge token; cache it for the next (re)connect.
    if (message.command === 'update_token') {
        if (boot) boot.token = message.token || '';
        return;
    }

    // From editor commands: put text in the chat box and optionally send it right away.
    // Sent by editor commands such as "Fix selection" (commands.js): the prompt arrives pre-built and the file path is
    // remembered so a later refactor result can be turned back into an edit on the same file.
    if (message.command === 'inject_chat') {
        const text = message.text || '';
        $('chatInput').value = text;
        currentTargetFilepath = message.filepath || '';
        switchWorkspace('chat');
        if (message.autoSend && text.trim()) {
            sendChatMessage();
        }
        return;
    }

    // Pre-fills the swarm objective only; the user still presses Run, because swarm runs write files.
    if (message.command === 'inject_swarm') {
        const objective = message.objective || '';
        if ($('objective')) $('objective').value = objective;
        switchWorkspace('swarm');
        return;
    }

    // Relay mode: frames and connection state come in as messages instead of through a local WebSocket.
    // ── Host-relay bridge events ──────────────────────────────────────────────
    // When KARL_USE_HOST_RELAY is active the extension host owns the WebSocket
    // and forwards frames/lifecycle events here via postMessage.
    // `socket_message` is the older name of the same relay event; both are accepted for compatibility.
    if (message.command === 'bridge_message' || message.command === 'socket_message') {
        try {
            // Frames may arrive as a JSON string or an already-parsed object.
            const data = typeof message.data === 'string'
                ? JSON.parse(message.data)
                : message.data;
            handleSocketMessage(data);
        } catch (err) {
            log(`[Relay] Malformed bridge frame: ${err.message}`);
        }
        return;
    }
    if (message.command === 'bridge_status') {
        const state = message.state || 'offline';
        if (state === 'connected') {
            lastConnectedAt = new Date();
            setConnectionState('connected', 'Connected');
            if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }

            // Immediately ask for the model/runtime status, then refresh every 4 s so the cockpit stays current.
            requestRuntimeStatus();
            loadAgentProfiles();
            runtimeStatusTimer = runtimeStatusTimer || setInterval(requestRuntimeStatus, 4000);
            persist();
        } else if (state === 'error') {
            lastBridgeError = message.message || 'Host relay error';
            handleDisconnect(true);
        } else {
            handleDisconnect(false);
        }
        return;
    }

    // Do not confuse with 'bridge_status': this one is host-side status, deliberately limited to outcomes the page cannot see.
    if (message.command === 'connection_state') {
        // Sent only for host-side connection outcomes the webview's own direct
        // socket can't observe itself (e.g. the host bridge's token being
        // rejected with close code 4001). Mirrors the same status-dot/label
        // update setConnectionState() already does for the direct-connect path.
        lastBridgeError = message.lastError || lastBridgeError;
        setConnectionState(message.state || 'offline', message.label || 'Offline');
        if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
        // Prevents an endless retry loop against a token the server already rejected.
        manualDisconnect = true; // Don't let the webview's own reconnect loop retry a rejected token
        return;
    }

    // Everything below updates the UI in response to host events: workflows, knowledge-base paths,
    // the review queue of proposed edits, swarm run history and cockpit state.
    // ── Webview ↔ host commands ───────────────────────────────────────────────
    // Chain of else-ifs rather than a switch because several branches share the same early-return style above;
    // each branch is a thin adapter from a host message to one render function.
    if (message.command === 'start_workflow') {
        startWorkflow(message.workflowId, message.data || {});
    } else if (message.command === 'set_kb_path') {
        $('kbPath').value = message.path || '';
        switchWorkspace('knowledge');
    } else if (message.command === 'search_kb_text') {
        $('kbQuery').value = message.query || '';
        switchWorkspace('knowledge');
        searchKb();
    } else if (message.command === 'open_review_bay') {
        switchWorkspace('changes');
    } else if (message.command === 'pending_file_edit') {
        addPendingEdit(message.edit);
    } else if (message.command === 'file_edit_previewed') {
        markPendingEdit(message.editId, 'previewed');
    } else if (message.command === 'file_edit_applied') {
        markPendingEdit(message.editId, 'applied');
    } else if (message.command === 'file_edit_status') {
        markPendingEdit(message.editId, message.status || 'proposed');
    } else if (message.command === 'file_edit_rejected') {
        removePendingEdit(message.editId, 'Rejected');
    } else if (message.command === 'file_edit_rolled_back') {
        markPendingEdit(message.editId, 'rolled_back');
    } else if (message.command === 'swarm_run_history') {
        renderSwarmRunHistory(message.runs || []);
    } else if (message.command === 'cockpit_state_update') {
        updateCockpitState(message.state || {});

    // Reports a backend crash (segfault or out-of-memory kill) as a chat bubble with advice.
    } else if (message.command === 'completion_diagnostic_failed') {
        _renderCompletionDiagnostic(message);
    }
});

/**
 * Attach every button/input handler once at start-up.
 *
 * Elements are guaranteed to exist because this runs after DOMContentLoaded. Lists that are re-rendered
 * often (review queue, sources, results...) use ONE delegated click handler on the container and read
 * the clicked button's data-* attributes; this avoids adding and leaking handlers on every re-render.
 */
function bindEvents() {
    // Top-level workspace tabs: data-workspace holds the id suffix of the matching <section id="workspace-...">.
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', () => switchWorkspace(btn.dataset.workspace));
    });

    // Sub-tabs inside a workspace: show only the matching ".subworkspace" panel.
    document.querySelectorAll('.subtab').forEach(btn => {
        btn.addEventListener('click', () => {
            // Scoping to the nearest workspace keeps sub-tabs of different workspaces independent.
            const container = btn.closest('.workspace');
            container.querySelectorAll('.subtab').forEach(sb => sb.classList.remove('active'));
            btn.classList.add('active');
            container.querySelectorAll('.subworkspace').forEach(sw => {
                const matches = sw.id === `subworkspace-${btn.dataset.subworkspace}`;
                sw.style.display = matches ? 'block' : 'none';
                sw.classList.toggle('active', matches);
            });
        });
    });

    // Connection, swarm, chat, knowledge-base and prompt-lab buttons.
    // Cockpit buttons
    // Both the cockpit and the settings workspace have connect buttons; they call the same functions so state stays consistent.
    $('cockpitConnectBtn').addEventListener('click', connect);
    $('cockpitDisconnectBtn').addEventListener('click', disconnect);
    $('clearRecentTasksBtn').addEventListener('click', () => {
        recentTasks = [];
        renderRecentTasks();
        persist();
    });

    $('connectBtn').addEventListener('click', connect);
    $('disconnectBtn').addEventListener('click', disconnect);
    $('runBtn').addEventListener('click', runSwarm);
    $('stopBtn').addEventListener('click', stopSwarm);
    $('askWorkspaceBtn').addEventListener('click', askWorkspace);
    $('refreshAgentProfilesBtn')?.addEventListener('click', loadAgentProfiles);
    $('agentProfileSelect')?.addEventListener('change', renderSelectedAgentProfile);
    $('saveAgentProfileBtn')?.addEventListener('click', saveSelectedAgentProfile);
    $('chatSendBtn').addEventListener('click', sendChatMessage);
    // Enter sends the message.
    $('chatInput').addEventListener('keydown', event => {
        // Enter sends the message immediately; there is no separate confirmation step.
        if (event.key === 'Enter') sendChatMessage();
    });
    $('refreshKbBtn').addEventListener('click', loadKbSources);
    $('activeFileKbBtn').addEventListener('click', () => vscode.postMessage({ command: 'use_active_file_for_kb' }));
    $('chooseKbFileBtn').addEventListener('click', () => vscode.postMessage({ command: 'choose_kb_file' }));
    $('chooseKbFolderBtn').addEventListener('click', () => vscode.postMessage({ command: 'choose_kb_folder' }));
    $('kbQueueAddBtn').addEventListener('click', addKbQueuePath);
    $('kbQueueRunBtn').addEventListener('click', ingestKbQueue);
    $('kbQueueClearBtn').addEventListener('click', () => {
        kbQueue = [];
        renderKbQueue();
    });
    $('kbIngestBtn').addEventListener('click', ingestKbPath);
    $('kbSearchBtn').addEventListener('click', searchKb);
    $('loadPairsBtn').addEventListener('click', loadPromptPairs);
    $('promptPairSelect').addEventListener('change', loadSelectedPromptPair);
    $('savePairBtn').addEventListener('click', savePromptPair);
    $('loadPairBtn').addEventListener('click', loadSelectedPromptPair);
    $('deletePairBtn').addEventListener('click', deletePromptPair);
    $('labRunBtn').addEventListener('click', runLab);
    $('diffBtn').addEventListener('click', computeLabDiff);
    // When "lock sync" is ticked, system prompt B mirrors A so both runs differ only in the user prompt.
    $('labSysA').addEventListener('input', () => {
        if ($('labLockSync').checked) $('labSysB').value = $('labSysA').value;
    });

    // requiresBridgeSupport() tells the user when the connected bridge does not offer the 'tokenize_text' method.
    $('tokenizeLabBtn').addEventListener('click', () => requiresBridgeSupport('Prompt Lab tokenizer visualization', 'tokenize_text'));
    $('loadModelsBtn').addEventListener('click', loadModels);
    $('loadAdaptersBtn').addEventListener('click', loadAdapters);
    // Adapters sit on top of a base model, so a base model filename must be known first.
    $('activateAdapterBtn').addEventListener('click', () => {
        if (!activeModelFilename) {
            vscode.postMessage({ command: 'show_error', text: 'No active baseline model is loaded yet. Select a GGUF model first.' });
            return;
        }
        const adapter = $('adapterSelect').value;

        // adapter: null (not undefined) so JSON keeps the key and the server clears any previous adapter.
        rpc(32, 'set_active_model', { 
            filename: activeModelFilename, 
            adapter: adapter || null 
        });
    });
    $('branchLatestBtn').addEventListener('click', branchFromLatest);
    $('newBranchBtn').addEventListener('click', createConversationBranch);
    
    // Git, diagnostics and vision buttons delegate to the host's runWorkflow command (commands.js).
    // Git Tab Actions
    // The workflow ids (reviewStagedDiff etc.) are defined in src/commands.js; the host gathers the diff itself, because the
    // webview has no access to git.
    $('gitReviewStagedBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'reviewStagedDiff' }));
    $('gitReviewUnstagedBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'reviewUnstagedDiff' }));
    $('gitReviewCombinedBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'reviewCombinedDiff' }));
    $('gitCommitMsgBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'generateCommitMessage' }));
    // Do not copy the placeholder text shown while the message is still being generated.
    $('gitCopyCommitBtn').addEventListener('click', async () => {
        const text = $('gitCommitText').innerText;
        if (text && text !== 'Generating commit message...') {
            await navigator.clipboard.writeText(text);
            vscode.postMessage({ command: 'show_message', text: 'Commit message copied to clipboard.' });
        }
    });

    // Diagnostics Tab Actions
    $('diagExplainFileBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'explainCurrentFileDiagnostics' }));
    $('diagExplainWorkspaceBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'explainDiagnostics' }));

    // Vision Tab Actions
    // Reuses the file-picker round trip of the knowledge base; the host answers with set_kb_path.
    $('visionSelectBtn').addEventListener('click', () => vscode.postMessage({ command: 'choose_kb_file' }));
    $('visionActiveBtn').addEventListener('click', () => vscode.postMessage({ command: 'runWorkflow', workflowId: 'analyzeImage' }));
    // The image must be chosen first; the path is passed to the workflow as its payload.
    $('visionAskBtn').addEventListener('click', () => {
        const path = $('visionImagePath').value.trim();
        if (!path) {
            vscode.postMessage({ command: 'show_error', text: 'Please select an image file first.' });
            return;
        }
        vscode.postMessage({ command: 'runWorkflow', workflowId: 'analyzeImage', payload: { filepath: path } });
    });
    $('visionErrBtn').addEventListener('click', () => {
        const path = $('visionImagePath').value.trim();
        if (!path) {
            vscode.postMessage({ command: 'show_error', text: 'Please select an image file first.' });
            return;
        }
        vscode.postMessage({ command: 'runWorkflow', workflowId: 'reviewScreenshotError', payload: { filepath: path } });
    });

    // Clear or copy all three log panes (swarm, training, eval).
    // Logs Actions
    // Resets each pane to its placeholder text instead of empty so the panes never look broken.
    $('clearLogsBtn').addEventListener('click', () => {
        $('terminal').innerText = '--- Swarm Logs ---';
        $('trainingLog').innerText = 'No training logs recorded.';
        $('evalLog').innerText = 'No eval execution logs.';
    });
    $('copyLogsBtn').addEventListener('click', async () => {
        const logs = `--- SWARM LOGS ---\n${$('terminal').innerText}\n\n--- TRAINING LOGS ---\n${$('trainingLog').innerText}\n\n--- EVAL LOGS ---\n${$('evalLog').innerText}`;
        await navigator.clipboard.writeText(logs);
        vscode.postMessage({ command: 'show_message', text: 'All logs copied to clipboard.' });
    });

    $('codexSearch').addEventListener('input', filterCodex);
    $('workspace').addEventListener('change', persist);
    // Changing the port while connected reconnects automatically to the new port.
    // Mirrors the new port into the cockpit label, saves it and (if online) reconnects so the user does not have to.
    $('bridgePort').addEventListener('change', () => {
        const port = $('bridgePort').value;
        $('cockpitPort').innerText = port;
        persist();
        if (isConnected() || connectionState === 'connecting') {
            log(`[Bridge] Port changed to ${port}. Reconnecting...`);
            disconnect();
            connect();
        }
    });
    $('taskMode').addEventListener('change', persist);
    
    // Every appearance control re-applies the whole look (applyAppearance also saves it).
    // Look workspace styling hooks
    $('themeSelect').addEventListener('change', applyAppearance);
    $('customAccent').addEventListener('input', applyAppearance);
    $('layoutSelect').addEventListener('change', applyAppearance);
    $('syncVsCodeTheme').addEventListener('change', applyAppearance);
    $('highContrastMode').addEventListener('change', applyAppearance);
    $('reducedMotion').addEventListener('change', applyAppearance);
    $('animationIntensity').addEventListener('input', applyAppearance);
    
    $('previewAllBtn').addEventListener('click', () => vscode.postMessage({ command: 'preview_all_files' }));
    $('rejectAllBtn').addEventListener('click', () => vscode.postMessage({ command: 'reject_all_files' }));
    $('rollbackSwarmRunBtn').addEventListener('click', () => vscode.postMessage({
        command: 'rollback_swarm_run',
        runId: $('swarmRunSelect').value
    }));
    $('copySummaryBtn').addEventListener('click', () => vscode.postMessage({ command: 'copy_patch_summary' }));
    $('changeFilter').addEventListener('change', renderPendingEdits);
    // Clearing keeps tasks that are still running.
    $('clearTasksBtn').addEventListener('click', () => {
        taskQueue = taskQueue.filter(task => task.status === 'running');
        renderTaskQueue();
    });
    
    // Defaults must match the defaults used by hydrate() in karl_state.js.
    // The literal defaults below must be kept in sync with hydrate() in karl_state.js.
    $('resetAppearanceBtn').addEventListener('click', () => {
        $('themeSelect').value = 'obsidian-core';
        $('customAccent').value = '#00c2ff';
        $('layoutSelect').value = 'cockpit';
        $('syncVsCodeTheme').checked = true;
        $('highContrastMode').checked = false;
        $('reducedMotion').checked = false;
        $('animationIntensity').value = 100;
        applyAppearance();
    });

    // One listener per container: buttons are looked up from event.target with closest(), so they work
    // even though the buttons are re-created on every render.
    // Delegated Event Listeners to prevent memory/event-listener leaks
    // Review Bay buttons: each carries one data-* attribute naming its action and the edit id, so a single listener on the
    // container handles preview, apply, reject, open, copy path and rollback.
    $('changeQueue').addEventListener('click', event => {
        const btn = event.target.closest('button');
        if (!btn) return;
        if (btn.dataset.preview) {
            vscode.postMessage({ command: 'preview_file', editId: btn.dataset.preview });
        } else if (btn.dataset.apply) {
            vscode.postMessage({ command: 'apply_file', editId: btn.dataset.apply });
        } else if (btn.dataset.reject) {
            vscode.postMessage({ command: 'reject_file', editId: btn.dataset.reject });
        } else if (btn.dataset.open) {
            vscode.postMessage({ command: 'open_file', editId: btn.dataset.open });
        } else if (btn.dataset.copyPath) {
            vscode.postMessage({ command: 'copy_file_path', editId: btn.dataset.copyPath });
        } else if (btn.dataset.rollback) {
            vscode.postMessage({ command: 'rollback_applied_file', editId: btn.dataset.rollback });
        }
    });

    // Line/character come from data attributes as strings, so they are converted to numbers for the host.
    $('diagnosticsList').addEventListener('click', event => {
        const el = event.target.closest('[data-diag-file]');
        if (!el) return;
        vscode.postMessage({
            command: 'open_diagnostic_line',
            filepath: el.dataset.diagFile,
            line: Number(el.dataset.diagLine),
            character: Number(el.dataset.diagChar)
        });
    });

    // Clicking a theme card only sets the hidden <select>; applyAppearance() reads that select as the single source of truth.
    $('themeGrid').addEventListener('click', event => {
        const card = event.target.closest('[data-theme-id]');
        if (!card) return;
        $('themeSelect').value = card.dataset.themeId;
        applyAppearance();
    });

    // Selecting a source doubles as a search filter, then the snapshot is re-rendered to highlight it.
    $('kbSourceList').addEventListener('click', event => {
        const row = event.target.closest('[data-source]');
        if (!row) return;
        kbSelectedSource = row.dataset.source || '';
        $('kbSourceFilter').value = kbSelectedSource;
        renderKbSnapshot(lastKbSnapshot);
    });

    $('kbQueue').addEventListener('click', event => {
        const btn = event.target.closest('[data-remove-kb]');
        if (!btn) return;
        kbQueue.splice(Number(btn.dataset.removeKb), 1);
        renderKbQueue();
    });

    $('quickActions').addEventListener('click', event => {
        const btn = event.target.closest('[data-workflow]');
        if (!btn) return;
        vscode.postMessage({
            command: 'runWorkflow',
            workflowId: btn.dataset.workflow,
            payload: {}
        });
    });

    // Re-run a recent task by sending its workflow id plus the saved objective and file.
    $('recentTasksHistory').addEventListener('click', event => {
        const btn = event.target.closest('[data-task-idx]');
        if (!btn) return;
        const task = recentTasks[Number(btn.dataset.taskIdx)];
        if (task) {
            vscode.postMessage({
                command: 'runWorkflow',
                workflowId: task.workflowId,
                payload: {
                    objective: task.objective,
                    filepath: task.filepath
                }
            });
        }
    });

    $('recentKbQueries').addEventListener('click', event => {
        const btn = event.target.closest('[data-kb-query]');
        if (!btn) return;
        $('kbQuery').value = recentKbQueries[Number(btn.dataset.kbQuery)] || '';
        searchKb();
    });

    // "Send to chat" and "Send to swarm" prefill the target input with a citation header plus the chunk text; they never submit.
    $('kbResults').addEventListener('click', event => {
        const btn = event.target.closest('button');
        if (!btn) return;
        if (btn.dataset.sendChat !== undefined) {
            const res = lastKbResults[Number(btn.dataset.sendChat)];
            if (res) {
                $('chatInput').value = `Reference: ${res.source_file} (Chunk ${res.chunk_id})\n\n${res.text}\n\n`;
                switchWorkspace('chat');
            }
        } else if (btn.dataset.sendSwarm !== undefined) {
            const res = lastKbResults[Number(btn.dataset.sendSwarm)];
            if (res) {
                $('objective').value = `Reference: ${res.source_file} (Chunk ${res.chunk_id})\n\n${res.text}\n\n${$('objective').value}`;
                switchWorkspace('swarm');
            }
        }
    });

    $('swarmRagSearchBtn').addEventListener('click', searchSwarmRag);
    $('swarmRagQuery').addEventListener('keydown', event => {
        if (event.key === 'Enter') searchSwarmRag();
    });
    // "Copy" copies the chunk text; "inject" prepends it as a reference in the swarm objective.
    $('swarmRagResults').addEventListener('click', event => {
        const btn = event.target.closest('button');
        if (!btn) return;
        const res = lastSwarmRagResults[Number(btn.dataset.idx)];
        if (!res) return;
        if (btn.dataset.copy !== undefined) {
            vscode.postMessage({ command: 'copy_to_clipboard', text: res.text || '' });
        } else if (btn.dataset.inject !== undefined) {
            const reference = `Reference: ${res.source_file} (Chunk ${res.chunk_id})\n\n${res.text}\n\n`;
            $('objective').value = `${reference}${$('objective').value}`;
        }
    });

    // Switch the active model (without an adapter).
    // The reply (id 32) is handled in karl_socket.js.
    $('modelList').addEventListener('click', event => {
        const btn = event.target.closest('[data-model]');
        if (!btn) return;
        rpc(32, 'set_active_model', { filename: btn.dataset.model });
    });

    $('codexList').addEventListener('click', event => {
        const row = event.target.closest('[data-topic]');
        if (!row) return;
        $('codexViewer').innerHTML = 'Loading reference...';
        rpc(21, 'get_codex_content', { topic: row.dataset.topic });
    });

    $('branchTree').addEventListener('click', event => {
        const btn = event.target.closest('[data-branch]');
        if (!btn) return;
        activeBranchId = btn.dataset.branch;
        renderBranches();
        persist();
    });

    // The viewer's content is HTML from the backend; only the two action buttons are handled here.
    $('codexViewer').addEventListener('click', event => {
        const btn = event.target.closest('button');
        if (!btn) return;
        if (btn.id === 'codexSendChatBtn') {
            const txt = $('codexViewer').querySelector('.codex-content').innerText;
            $('chatInput').value = `Codex Reference:\n\n${txt}\n\n`;
            switchWorkspace('chat');
        } else if (btn.id === 'codexSendSwarmBtn') {
            const txt = $('codexViewer').querySelector('.codex-content').innerText;
            $('objective').value = `Codex Reference:\n\n${txt}\n\n${$('objective').value}`;
            switchWorkspace('swarm');
        }
    });
}

/**
 * Handle the host's ``start_workflow`` message: prefill the UI for the chosen workflow and open the target tab.
 *
 * The tab decides what happens next: git/diagnostics/vision tabs show their result panels, chat auto-sends
 * the prompt after 100 ms, and the swarm tab connects (if needed) and starts a run.
 * @param {string} workflowId
 * @param {any} data
 */
function startWorkflow(workflowId, data) {
    // Remembered so the response handler can route the result (for example a commit message) to the right panel.
    activeWorkflowId = workflowId;
    $('taskMode').value = workflowId;
    $('workspace').value = data.workspace_path || '';
    $('objective').value = data.objective || '';
    currentTargetFilepath = data.filepath || '';
    
    // Shows what context (file, selection, diagnostics) the host attached, so the user can judge the prompt.
    renderContextMeta(data.context_meta);
    rememberTask(workflowId, data.mode || 'Task', data.objective || '', data.filepath || '');
    
    const tab = data.targetTab || 'swarm';
    switchWorkspace(tab);

    // Each branch below prepares the placeholder UI of its tab; the real content arrives later as chat/swarm output.
    if (tab === 'git') {
        $('gitDiffContainer').style.display = 'block';
        $('gitDiffText').innerText = data.code || 'No changes or diff found.';
        if (workflowId === 'generateCommitMessage') {
            $('gitCommitOutput').style.display = 'block';
            $('gitCommitText').innerText = 'Generating commit message...';
        }
    }

    if (tab === 'diagnostics') {
        $('diagExplanation').style.display = 'block';
        $('diagExplanationText').innerText = 'Karl is reviewing problem diagnostics...';
    }

    if (tab === 'vision') {
        $('visionImagePath').value = data.filepath || '';
        $('visionPreviewCard').style.display = 'block';

        // escapeHtml is required: the file name comes from the filesystem and is inserted as HTML.
        $('visionImagePreview').innerHTML = `<span style="color:var(--karl-accent-2); font-weight:700;">Image: ${escapeHtml(data.filepath.split('/').pop())}</span>`;
        $('visionResult').style.display = 'block';
        $('visionResultText').innerText = 'Karl is inspecting image file...';
    }

    // The short delay lets the tab finish switching before the message is sent.
    if (tab === 'chat') {
        const chatInput = $('chatInput');
        if (chatInput && data.prompt) {
            chatInput.value = data.prompt;
            setTimeout(() => {
                sendChatMessage();
            }, 100);
        }
    }

    if (tab === 'swarm') {
        // Swarm workflows start immediately (unlike chat, which also auto-sends) because the host already built the objective.
        connectAndRun();
    }
}

/**
 * Run the swarm on the typed objective, or on a default "review this workspace" objective if empty.
 */
function askWorkspace() {
    const objective = $('objective').value.trim() || 'Review this workspace and identify the highest value next improvements.';
    $('objective').value = objective;
    runSwarm();
}

/**
 * Start a swarm run, connecting to the bridge first if necessary (gives up after 8 seconds).
 */
function connectAndRun() {
    if (isConnected()) {
        runSwarm();
        return;
    }
    connect();
    const started = Date.now();
    // Poll every 150 ms until the connection is up or the 8 s deadline passes.
    // Polling is simpler than wiring a one-shot "connected" event, and bounded by the 8 s deadline.
    const timer = setInterval(() => {
        if (isConnected()) {
            clearInterval(timer);
            runSwarm();
        } else if (Date.now() - started > 8000) {
            clearInterval(timer);
            vscode.postMessage({ command: 'show_error', text: 'Karl bridge did not connect within 8 seconds.' });
        }
    }, 150);
}

/**
 * Validate the form and send the ``submit_task`` request (rpc id 1) to start the swarm.
 *
 * Needs an objective and a workspace path. If the bridge is down it connects first. Settings such as timeout
 * (default 600 s), max loops (default 5), auto-steering and the three agent profiles are sent as hyperparams.
 */
function runSwarm() {
    const objective = $('objective').value.trim();
    const workspace = $('workspace').value.trim();
    const testCommand = $('testCmd').value.trim();
    if (!objective || !workspace) {
        vscode.postMessage({ command: 'show_error', text: 'Objective and workspace path are required.' });
        return;
    }
    if (!isConnected()) {
        connectAndRun();
        return;
    }

    // Visually marks the bridge as busy until a finished_swarm notification resets it.
    setConnectionState('running', 'Running');
    // Track the run in the task queue and in the recent-task history.
    activeTaskId = addTask($('taskMode').value, objective);
    rememberTask($('taskMode').value, $('taskMode').selectedOptions?.[0]?.text || 'Task', objective, workspace);
    // Reset the visible swarm output so logs from a previous run do not mix with this one.
    $('terminal').innerText = '--- Swarm Logs ---';
    // The timeline is rebuilt by incoming progress events (task plan, layer, file edited...).
    $('timeline').innerHTML = '';
    // First timeline entry records which task mode (e.g. general/refactor) was launched.
    addTimeline('Launch', $('taskMode').value);
    log('[Swarm] Deploying Karl agents.');
    // rpc id 1 is reserved for submit_task; karl_socket.js uses the id to recognise the reply.
    rpc(1, 'submit_task', {
        objective,
        workspace_path: workspace,
        // Empty string means "no command": the Tester stage then skips execution rather than guessing.
        test_command: testCommand,
        // hyperparams() (karl_state.js) supplies shared sampling settings; the swarm-specific limits
        // below override or extend them for this one task.
        hyperparams: {
            ...hyperparams(),
            // `|| 600` / `|| 5` cover empty or non-numeric inputs (Number('') is 0, Number('x') is NaN).
            timeout_seconds: Number($('swarmTimeout').value) || 600,
            max_loops: Number($('swarmMaxLoops').value) || 5,
            // Boolean: when true the orchestrator may auto-inject corrective guidance (drift detector).
            auto_steering_enabled: $('swarmAutoSteer').checked,
            // Profile ids per stage; the server maps them to model/adapter/system-prompt overrides.
            agent_profiles: {
                architect: $('swarmArchitectProfile').value || 'architect',
                coder: $('swarmCoderProfile').value || 'coder',
                tester: $('swarmTesterProfile').value || 'tester'
            }
        }
    });
}

/**
 * Ask the backend to stop the running swarm task (rpc id 2).
 */
function stopSwarm() {
    // No params: the server stops whatever task this connection started.
    rpc(2, 'stop_task');
}

/**
 * Request the available agent profiles from the backend (rpc id 70).
 */
function loadAgentProfiles() {
    // Empty params object keeps the JSON-RPC payload well-formed ("params": {}).
    rpc(70, 'get_agent_profiles', {});
}

/**
 * Fill all profile dropdowns, keeping the user's current choice or falling back to the architect/coder/tester defaults.
 */
function renderAgentProfileOptions() {
    // swarmAgentProfiles is a map id -> profile filled by the get_agent_profiles reply.
    const entries = Object.entries(swarmAgentProfiles || {});
    // One shared <option> list is rendered for every dropdown, so all selects offer the same profiles.
    const optionHtml = entries.map(([id, profile]) => {
        const label = `${profile.icon || ''} ${profile.name || id}`.trim();
        // escapeHtml (karl_render.js) is mandatory: profile names are user-editable text.
        return `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`;
    }).join('');
    // Preferred defaults when nothing was chosen yet or the old choice no longer exists.
    const defaultSelections = {
        swarmArchitectProfile: 'architect',
        swarmCoderProfile: 'coder',
        swarmTesterProfile: 'tester'
    };
    // The three swarm stage selectors plus the editor's own selector all show the same profile list.
    ['swarmArchitectProfile', 'swarmCoderProfile', 'swarmTesterProfile', 'agentProfileSelect'].forEach(id => {
        const el = $(id);
        // Some dropdowns may not exist if the HTML variant omits a panel, so skip silently.
        if (!el) return;
        // Remember the user's selection across the re-render.
        const previous = el.value;
        el.innerHTML = optionHtml || '<option value="">No profiles loaded</option>';
        // Keep the previous choice only if that profile still exists after the reload.
        if (previous && entries.some(([profileId]) => profileId === previous)) {
            el.value = previous;
        } else if (defaultSelections[id] && swarmAgentProfiles[defaultSelections[id]]) {
            el.value = defaultSelections[id];
        }
    });
    // Refresh the edit form and spec card to match whatever is now selected.
    renderSelectedAgentProfile();
}

/**
 * Copy the selected profile (name, icon, prompt, temperature, context limit...) into the edit form.
 */
function renderSelectedAgentProfile() {
    // Optional chaining: the select may be missing in a minimal layout.
    const id = $('agentProfileSelect')?.value || '';
    const profile = swarmAgentProfiles[id] || null;
    // No selection: show a hint and leave the form fields as they were.
    if (!profile) {
        $('agentProfileSpec').textContent = 'No profile selected.';
        return;
    }
    $('agentProfileName').value = profile.name || id;
    $('agentProfileIcon').value = profile.icon || '';
    $('agentProfilePrompt').value = profile.system_prompt || profile.prompt || '';
    // `??` rather than `||` so a legitimate temperature of 0 is not replaced by the default.
    $('agentProfileTemp').value = profile.temperature ?? 0.2;
    $('agentProfileContext').value = profile.context_limit ?? 2048;
    // Tool permissions are stored as booleans: read files, write files, run code in the sandbox, query RAG.
    // Tool permissions are stored as booleans: read files, write files, run code in the sandbox, query RAG.
    const tools = profile.tools || {};
    $('agentToolRead').checked = !!tools.read_files;
    $('agentToolWrite').checked = !!tools.write_files;
    $('agentToolExec').checked = !!tools.execute_sandbox;
    $('agentToolRag').checked = !!tools.query_rag;
    // The spec card is a read-only summary; innerHTML is safe here because every value is escaped.
    $('agentProfileSpec').innerHTML = `
        <div><strong>${escapeHtml(profile.name || id)}</strong> <span class="muted">${escapeHtml(id)}</span></div>
        <div class="muted">Temp ${Number(profile.temperature ?? 0).toFixed(2)} · Context ${profile.context_limit || 2048} · ${profile.builtin ? 'built-in' : 'custom'}</div>
    `;
}

/**
 * Send the edited profile form to the backend (rpc id 71 ``save_agent_profile``).
 *
 * Blank name/icon fall back to the id; a missing context limit falls back to 2048. The reply triggers
 * a reload of all profiles (see handleRpcResult in karl_socket.js).
 */
function saveSelectedAgentProfile() {
    const id = $('agentProfileSelect')?.value || '';
    // The Save button is meaningless without a selected profile, so surface an error via the host,
    // which shows it as a VS Code notification.
    if (!id) {
        vscode.postMessage({ command: 'show_error', text: 'Select an agent profile first.' });
        return;
    }
    // rpc id 71 pairs with the "save_agent_profile" reply handled in karl_socket.js.
    rpc(71, 'save_agent_profile', {
        profile: {
            id,
            name: $('agentProfileName').value.trim() || id,
            // Fall back to the first letter of the id so the profile always has some icon.
            icon: $('agentProfileIcon').value.trim() || id.slice(0, 1).toUpperCase(),
            system_prompt: $('agentProfilePrompt').value,
            // Number('') is 0 and Number('x') is NaN; `|| 0` maps both to a usable temperature.
            temperature: Number($('agentProfileTemp').value) || 0,
            // 2048 tokens is the smallest context the backend accepts for a profile.
            context_limit: Number($('agentProfileContext').value) || 2048,
            // Tool permissions: each box becomes a strict boolean the Python side checks before a tool runs.
            tools: {
                read_files: $('agentToolRead').checked,
                write_files: $('agentToolWrite').checked,
                execute_sandbox: $('agentToolExec').checked,
                query_rag: $('agentToolRag').checked
            }
        }
    });
}

// ── thoughts panel initialisation ─────────────────────────────────────────────
// Transforms the static HTML inside #introspectionBox into a collapsible
// reasoning panel matching Karl's Obsidian aesthetic. Called once on load.
// Detail: replaces the plain label with a clickable header (token count + pulse dot) and makes the
// text body collapsible by toggling max-height/padding.
function _initThoughtsPanel() {
    // Both elements come from the static HTML in sidebarProvider.js; exit quietly if the layout lacks them.
    const box = $('introspectionBox');
    const thoughts = $('introspectionThoughts');
    if (!box || !thoughts) return;

    // Replace the static .eyebrow div with a full interactive header
    // Keep a reference to the original label so it can be swapped for the interactive header.
    const oldEyebrow = box.querySelector('.eyebrow');
    const header = document.createElement('div');
    // These ids are used by updateThoughts()/finalizeThoughts() (karl_render.js) to update the token
    // counter and stop the pulse dot.
    header.id = 'thoughtsHeader';
    header.className = 'thoughts-header';
    header.innerHTML = `
        <span class="thoughts-toggle" id="thoughtsToggle">▼</span>
        <span class="eyebrow" style="margin:0">Reasoning</span>
        <span id="thoughtsTokenCount" class="thoughts-token-count">0 tokens</span>
        <span id="thoughtsPulseDot" class="thoughts-pulse-dot"></span>
    `;
    // replaceChild keeps the header at the same position the old label occupied.
    if (oldEyebrow) {
        box.replaceChild(header, oldEyebrow);
    } else {
        // No old label found: put the header before the reasoning text instead.
        box.insertBefore(header, thoughts);
    }

    // Mark the <pre> so CSS targets it as the scrollable body
    // Mark the <pre> so CSS targets it as the scrollable body
    thoughts.classList.add('thoughts-body');

    // Wire the collapse/expand toggle
    // Collapse is done with max-height 0 and zero padding (not display:none) so that the CSS height
    // transition can animate and streaming text keeps being appended while hidden.
    // Collapse state lives in this closure; clicking the header flips it and updates the arrow.
    let _collapsed = false;
    // One listener on the header toggles both states.
    header.addEventListener('click', () => {
        _collapsed = !_collapsed;
        if (_collapsed) {
            thoughts.style.maxHeight = '0';
            thoughts.style.paddingTop = '0';
            thoughts.style.paddingBottom = '0';
        } else {
            // Empty strings remove the inline style, handing control back to the stylesheet.
            thoughts.style.maxHeight = '';
            thoughts.style.paddingTop = '';
            thoughts.style.paddingBottom = '';
        }
        // Arrow glyphs: right-pointing = collapsed, down-pointing = expanded.
        $('thoughtsToggle').textContent = _collapsed ? '▶' : '▼';
        // The modifier class lets CSS also restyle the header while collapsed.
        header.classList.toggle('thoughts-header--collapsed', _collapsed);
    });
}

// Module-level so handleChatFinished() can read what the stream produced after the bubbles are gone.
// Raw text of the response being streamed (including any <think> markup); used to extract refactored code.
let _lastAssistantContent = '';

/**
 * Send the chat box text to the backend (rpc id 3 ``submit_chat``) and prepare the streaming bubbles.
 *
 * Ignored when the box is empty or a reply is still streaming. Adds the user bubble and an empty assistant
 * bubble that tokens will fill, then locks the send button until the reply finishes.
 */
function sendChatMessage() {
    const input = $('chatInput');
    const text = input.value.trim();
    // Two guards: nothing to send, or a reply is still streaming (chatFinished false). The second
    // guard prevents interleaved streams in the single shared bubble.
    if (!text || !chatFinished) return;
    // Fail fast with a visible message instead of letting rpc() silently drop the request.
    if (!isConnected()) {
        vscode.postMessage({ command: 'show_error', text: 'Karl is disconnected.' });
        return;
    }
    // The file the user is working on (from the cockpit status) becomes the target for refactor results.
    currentTargetFilepath = currentTargetFilepath || $('cockpitFile').title || '';
    // The user bubble appears immediately; the assistant bubble starts empty and is filled by
    // handleResponseToken().
    appendMessageBubble('user', text);
    appendMessageBubble('assistant', '');
    // Saved so recordConversationTurn() can pair the answer with its question.
    currentConversationInput = text;
    _lastAssistantContent = '';
    // Remember what was sent: handleChatFinished() checks whether it started with "refactor".
    $('chatInput').dataset.lastSent = $('chatInput').value;
    input.value = '';
    chatFinished = false;
    // CSS hook: lets styles show a "generating" state.
    document.body.dataset.wsState = 'generating';
    $('chatSendBtn').disabled = true;
    // Reset the reasoning panel for this new generation
    // Reset the reasoning panel for this new generation
    resetThoughtsPanel();
    currentResponseText = '';
    // rpc id 3 pairs with submit_chat; tokens then arrive as notifications, not as a reply.
    rpc(3, 'submit_chat', {
        message: text,
        // workspace_path lets the server resolve relative file mentions and codebase memory.
        workspace_path: $('workspace').value.trim(),
        hyperparams: hyperparams()
    });
}

/**
 * Route one streamed response token to its destination.
 *
 * Prompt-lab runs fill output A or B; git/diagnostics/vision workflows show the text in their own panel;
 * normal chat appends to the assistant bubble. The accumulated text is also kept for later use.
 * @param {string} token
 */
function handleResponseToken(token) {
    // Two accumulators: one for refactor extraction (raw, with think tags) and one for per-workflow panels.
    _lastAssistantContent += token;
    currentResponseText += token;
    
    // Prompt-lab mode takes priority: tokens go to the A/B boxes, not the chat.
    // Prompt-lab mode takes priority: tokens go to the A/B boxes, not the chat.
    // Lab runs reuse the chat stream; labRunning tells this router to divert instead of rendering a bubble.
    if (labRunning) {
        if (currentLabTarget === 'A') {
            labOutputA += token;
            $('labOutputA').innerText = labOutputA;
        } else {
            labOutputB += token;
            $('labOutputB').innerText = labOutputB;
        }
        return;
    }

    // Workflow helpers (git commit message, diagnostics, vision) mirror the stream in their own panel
    // in addition to the chat bubble (the append below still runs).
    if (activeWorkflowId === 'generateCommitMessage') {
        $('gitCommitText').innerText = currentResponseText;
    } else if (activeWorkflowId === 'explainDiagnostics' || activeWorkflowId === 'explainCurrentFileDiagnostics') {
        $('diagExplanationText').innerText = currentResponseText;
    } else if (activeWorkflowId === 'analyzeImage' || activeWorkflowId === 'reviewScreenshotError') {
        $('visionResultText').innerText = currentResponseText;
    }

    // Normal chat rendering: appends to the open bubble using the cheap streaming path (no re-parse).
    appendChatToken(token);
}

/**
 * Finish a generation.
 *
 * In prompt-lab mode it chains A then B and finally computes the diff. Otherwise it removes the cursor,
 * re-enables sending, records the turn in the conversation branch and, for "refactor ..." prompts, sends the
 * first code block of the answer to the host so the editor can apply it.
 */
function handleChatFinished() {
    // A and B share one connection and run serially; finishing A starts B, finishing B ends the lab run.
    if (labRunning) {
        if (currentLabTarget === 'A') {
            currentLabTarget = 'B';
            $('labOutputB').innerText = 'Generating output B...';
            sendLabMessage('B', $('labSysB').value, $('labUser').value);
        } else {
            computeLabDiff();
        }
        return;
    }
    // Remove streaming cursor and stop the pulse dot before recording the turn
    // Remove streaming cursor and stop the pulse dot before recording the turn
    removeStreamingCursor();
    finalizeThoughts();
    // Unlock sending only now: guarantees the previous reply is complete before the next submit.
    chatFinished = true;
    $('chatSendBtn').disabled = false;
    // Return the body state to whatever the connection dot says (connected/busy/...).
    // Return the body state to whatever the connection dot says (connected/busy/...).
    // Restore ws-state indicator (use statusDot class as source of truth)
    const _dot = $('statusDot');
    if (_dot) {
        const _st = [..._dot.classList].find(c => c !== 'status-dot') || 'connected';
        document.body.dataset.wsState = _st;
    }
    // Read the rendered text (not the raw stream) so the stored turn matches what the user saw.
    const lastAssistant = $('chatMessages').querySelector('.message.assistant:last-child .message-content');
    recordConversationTurn(currentConversationInput, lastAssistant ? lastAssistant.innerText : '');
    // Clear per-turn state so a stray notification cannot attach to the next turn.
    currentConversationInput = '';
    activeWorkflowId = '';

    // Check if this was a refactor request — extract code block and notify host
    // Check if this was a refactor request — extract code block and notify host
    const lastInput = $('chatInput').dataset.lastSent || '';
    // Refactor results are applied by the host (``refactor_result`` message), not by this page.
    if (lastInput.toLowerCase().startsWith('refactor')) {
        const fullResponse = _lastAssistantContent || '';
        // Matches the first fenced block (```lang\n ... ```); the lazy group stops at the first closing fence.
        const codeMatch = fullResponse.match(/```[\w]*\n([\s\S]*?)```/);
        if (codeMatch) {
            // The extension host applies it to the editor; this page never edits files itself.
            vscode.postMessage({
                command: 'refactor_result',
                code: codeMatch[1],
                filepath: currentTargetFilepath
            });
        }
    }
}

/**
 * Return the active conversation branch, creating one if none exists yet.
 */
function activeBranch() {
    // Lazily create the first branch so recording a turn never fails on a fresh install.
    if (!activeBranchId) createConversationBranch();
    // Fall back to the first branch if the id is stale (e.g. after a persisted-state schema change).
    return conversationBranches.find(branch => branch.id === activeBranchId) || conversationBranches[0];
}

/**
 * Create a new empty conversation branch at the top of the list, make it active and save.
 * @param {string} [seedTitle]
 */
function createConversationBranch(seedTitle = '') {
    // Unique enough for client-side ids: millisecond time plus random hex; ids are only compared locally.
    const id = `branch-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const branch = {
        id,
        title: seedTitle || `Branch ${conversationBranches.length + 1}`,
        turns: [],
        createdAt: new Date().toISOString()
    };
    // unshift puts the newest branch first in the sidebar list.
    conversationBranches.unshift(branch);
    activeBranchId = id;
    // renderBranches updates the list; persist() writes webview state so branches survive a reload.
    renderBranches();
    persist();
}

/**
 * Fork the active branch, dropping its last turn (so the user can retry from just before the latest answer).
 */
function branchFromLatest() {
    // slice(0, -1) copies all turns except the last one, so the fork re-asks the final question.
    const current = activeBranch();
    const id = `branch-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const branch = {
        id,
        title: `Fork of ${current.title || 'Workbench'}`,
        turns: current.turns.slice(0, -1),
        createdAt: new Date().toISOString()
    };
    conversationBranches.unshift(branch);
    activeBranchId = id;
    renderBranches();
    // Tell the user where the new branch appears.
    vscode.postMessage({ command: 'show_message', text: 'Karl conversation branch created in the Chat panel.' });
    persist();
}

/**
 * Store a finished user/assistant exchange in the active branch (the branch title becomes the first 48 characters of the prompt).
 *
 * The branch moves to the front; at most 20 branches are kept.
 * @param {string} user
 * @param {string} assistant
 */
function recordConversationTurn(user, assistant) {
    // Coerce to string: callers may pass undefined when a stream was cancelled.
    const prompt = String(user || '').trim();
    const response = String(assistant || '').trim();
    // Nothing to record (cancelled before any text).
    if (!prompt && !response) return;
    const branch = activeBranch();
    branch.turns.push({ user: prompt, assistant: response, at: new Date().toISOString() });
    // Titles are the first 48 characters of the question; keep the old title for empty prompts.
    branch.title = prompt.slice(0, 48) || branch.title;
    // Move the active branch to the front and cap the history at 20 so persisted state stays small.
    conversationBranches = [branch, ...conversationBranches.filter(item => item.id !== branch.id)].slice(0, 20);
    activeBranchId = branch.id;
    renderBranches();
    persist();
}

/**
 * Add (or replace) a proposed file edit in the review queue and redraw it.
 * @param {any} edit
 */
function addPendingEdit(edit) {
    // Map keyed by edit id preserves insertion order for rendering and makes replace-by-id trivial.
    if (!edit || !edit.id) return;
    // Same id re-sent by the server updates the existing entry instead of duplicating it.
    pendingEdits.set(edit.id, edit);
    renderPendingEdits();
}

/**
 * Set an edit's status (previewed, applied, rolled_back...) and redraw the queue.
 * @param {string} editId
 * @param {string} status
 */
function markPendingEdit(editId, status) {
    const edit = pendingEdits.get(editId);
    // Edit objects are shared references, but set() keeps the Map and any future cloning logic consistent.
    if (edit) {
        edit.status = status;
        pendingEdits.set(editId, edit);
        renderPendingEdits();
    }
}

/**
 * Mark the first approved/submitted edit for ``filepath`` with ``status`` (used when the backend reports a write).
 * @param {string} filepath
 * @param {string} status
 */
function markPendingEditByPath(filepath, status) {
    // Matching by path is a fallback for server events that report a write without an edit id.
    if (!filepath) return;
    for (const [editId, edit] of pendingEdits.entries()) {
        // Only edits already approved/submitted can be written, so unrelated pending edits are skipped.
        if (edit.filepath === filepath && ['approved', 'submitted'].includes(edit.status)) {
            markPendingEdit(editId, status);
            return;
        }
    }
}

/**
 * Fill the rollback dropdown with the 5 most recent swarm runs.
 * @param {any[]} runs
 */
function renderSwarmRunHistory(runs) {
    // Show only the 5 latest runs; older ones remain on disk but the dropdown stays short.
    swarmRunHistory = Array.isArray(runs) ? runs.slice(0, 5) : [];
    const select = $('swarmRunSelect');
    if (!select) return;
    // Empty state text so the Rollback button has an explanation instead of a blank list.
    if (!swarmRunHistory.length) {
        select.innerHTML = '<option value="">No swarm runs yet</option>';
        return;
    }
    select.innerHTML = swarmRunHistory.map(run => {
        // Locale string is for display only; the real timestamp stays in the history array.
        const when = run.timestamp ? new Date(run.timestamp).toLocaleString() : 'unknown time';
        const files = Array.isArray(run.changedFiles) ? run.changedFiles.length : 0;
        const objective = (run.objective || run.summary || 'Swarm run').slice(0, 48);
        // value is the runId that the rollback RPC needs; everything else is label text, escaped.
        return `<option value="${escapeHtml(run.runId || '')}">${escapeHtml(when)} · ${escapeHtml(objective)} · ${files} file(s)</option>`;
    }).join('');
}

/**
 * Remove an edit from the queue (e.g. rejected) and log it.
 * @param {string} editId
 * @param {string} state label used in the log line
 */
function removePendingEdit(editId, state) {
    // Removing instead of marking keeps rejected edits out of the review queue.
    pendingEdits.delete(editId);
    renderPendingEdits();
    log(`[Changes] ${state} pending edit ${editId}.`);
}

/**
 * Start a Prompt Lab comparison: run prompt A first; handleChatFinished() then runs B and the diff.
 *
 * Needs a user message and a live connection; a second click while running is ignored.
 */
function runLab() {
    // Re-entrancy guard: a second click while A/B is already running would start parallel streams.
    if (labRunning) return;
    if (!$('labUser').value.trim()) {
        vscode.postMessage({ command: 'show_error', text: 'Prompt Lab needs a user message.' });
        return;
    }
    if (!isConnected()) {
        vscode.postMessage({ command: 'show_error', text: 'Karl is disconnected.' });
        return;
    }
    // Switch to "running" state: lock the Run button and reset both outputs, so stale text from an
    // earlier comparison can never be mistaken for the new result.
    labRunning = true;
    currentLabTarget = 'A';
    // Raw (unrendered) outputs are accumulated here by handleResponseToken(); the diff uses them.
    labOutputA = '';
    labOutputB = '';
    $('labRunBtn').disabled = true;
    $('labOutputA').innerText = 'Generating output A...';
    $('labOutputB').innerText = 'Waiting...';
    $('labDiff').innerText = 'Waiting for both runs...';
    // Starts with A; handleChatFinished() will call sendLabMessage('B', ...) when A completes.
    sendLabMessage('A', $('labSysA').value, $('labUser').value);
}

/**
 * Submit one lab run. The rpc id is 10 for A and 11 for B so replies can be told apart.
 * @param {string} target "A" or "B"
 * @param {string} systemPrompt
 * @param {string} userMsg
 */
function sendLabMessage(target, systemPrompt, userMsg) {
    // currentLabTarget tells handleResponseToken which output box receives the incoming tokens.
    currentLabTarget = target;
    // Shared sampling settings, extended with this side's own system prompt (A and B differ only there).
    const params = hyperparams();
    params.system_prompt = systemPrompt;
    // Ids 10 and 11 let karl_socket.js tell the A reply from the B reply.
    rpc(10 + (target === 'A' ? 0 : 1), 'submit_chat', {
        message: userMsg,
        hyperparams: params
    });
}

/**
 * Render the diff of the two lab outputs.
 */
function computeLabDiff() {
    // renderLabDiff (karl_render.js) shows a word/line diff with additions and deletions highlighted.
    renderLabDiff(labOutputA, labOutputB);
}

/**
 * Ask the backend for the selected (or typed) saved prompt pair (rpc id 41).
 */
function loadSelectedPromptPair() {
    // The dropdown wins; otherwise the typed name is used so a pair can be loaded by typing.
    const name = $('promptPairSelect').value || $('promptPairName').value.trim();
    // rpc id 41 -> get_prompt_pair; the reply is handed to applyPromptPair().
    if (name) rpc(41, 'get_prompt_pair', { name });
}

/**
 * Save the current lab setup and outputs under the given name (rpc id 42).
 *
 * Both sides currently share the same user message and the same RAG/loop checkboxes.
 */
function savePromptPair() {
    const name = $('promptPairName').value.trim();
    if (!name) {
        vscode.postMessage({ command: 'show_error', text: 'Prompt pair name is required.' });
        return;
    }
    // Both sides intentionally use the same user message today (one shared textbox).
    const userMsg = $('labUser').value;
    // rpc id 42 -> save_prompt_pair. The server stores the JSON under data/prompt_pairs/<name>.json.
    rpc(42, 'save_prompt_pair', {
        name,
        system_a: $('labSysA').value,
        user_a: userMsg,
        system_b: $('labSysB').value,
        user_b: userMsg,
        rag_a: $('karlRag').checked,
        rag_b: $('karlRag').checked,
        loop_a: $('karlLoop').checked,
        loop_b: $('karlLoop').checked,
        // Both the raw stream and the displayed text are saved, so loading shows exactly what was seen.
        output_a_raw: labOutputA,
        output_b_raw: labOutputB,
        output_a_display: $('labOutputA').innerText,
        output_b_display: $('labOutputB').innerText
    });
}

/**
 * Delete the selected (or typed) saved prompt pair (rpc id 43).
 */
function deletePromptPair() {
    const name = $('promptPairSelect').value || $('promptPairName').value.trim();
    // rpc id 43 -> delete_prompt_pair.
    if (name) rpc(43, 'delete_prompt_pair', { name });
}

/**
 * Fill the lab form (prompts, toggles, saved outputs) from a loaded pair and redraw the diff.
 * @param {any} result
 */
function applyPromptPair(result) {
    // Defensive: server replies may be null on error.
    if (!result || typeof result !== 'object') return;
    if (result.name) $('promptPairName').value = result.name;
    // `!== undefined` checks (not truthiness) let an empty string legitimately clear a field.
    if (result.system_a !== undefined) $('labSysA').value = result.system_a;
    if (result.user_a  !== undefined) $('labUser').value  = result.user_a;
    if (result.system_b !== undefined) $('labSysB').value = result.system_b;
    if (result.rag_a   !== undefined) $('karlRag').checked  = !!result.rag_a;
    if (result.loop_a  !== undefined) $('karlLoop').checked = !!result.loop_a;
    // Outputs reset to '' when the saved pair has none, so no stale output survives a load.
    labOutputA = result.output_a_raw || '';
    labOutputB = result.output_b_raw || '';
    // Only redraw the diff when there is something to compare.
    if (labOutputA || labOutputB) {
        $('labOutputA').innerText = result.output_a_display || labOutputA;
        $('labOutputB').innerText = result.output_b_display || labOutputB;
        renderLabDiff(labOutputA, labOutputB);
    }
    log(`[Prompt Lab] Loaded pair: ${result.name || '(unnamed)'}`);
}

/**
 // Queue invariants: newest first, one entry per path, at most 50 so persisted state stays small.
 * Add the path in the knowledge-base path box to the ingest queue (newest first, no duplicates, max 50).
 */
function addKbQueuePath() {
    const ingestPath = $('kbPath').value.trim();
    // Empty path: nothing to queue; tell the user instead of silently ignoring the click.
    if (!ingestPath) {
        vscode.postMessage({ command: 'show_error', text: 'Choose a file or folder before adding it to the queue.' });
        return;
    }
    // filter() removes an older entry for the same path, which moves it to the front with status 'queued'.
    kbQueue = [{ path: ingestPath, status: 'queued' }, ...kbQueue.filter(item => item.path !== ingestPath)].slice(0, 50);
    renderKbQueue();
}

/**
 * Start processing the knowledge-base queue (adds the typed path first if the queue is empty).
 */
function ingestKbQueue() {
    // Convenience: pressing Ingest with an empty queue ingests whatever path is typed.
    if (!kbQueue.length) {
        addKbQueuePath();
    }
    // kbQueueRunning prevents starting a second concurrent ingest while one is still streaming results.
    if (!kbQueue.length || kbQueueRunning) return;
    ingestNextQueuedPath();
}

/**
 * Start ingesting the next queued path; clears the running flag when none are left.
 */
function ingestNextQueuedPath() {
    const next = kbQueue.find(item => item.status === 'queued');
    // Queue drained: reset flags and refresh the source list so new documents appear.
    if (!next) {
        kbQueueRunning = false;
        renderKbQueue();
        loadKbSources();
        return;
    }
    kbQueueRunning = true;
    // Mark running before the RPC so the UI shows progress immediately.
    next.status = 'running';
    // ingestKbPath reads the path from the textbox, so the queued path is copied there first.
    $('kbPath').value = next.path;
    renderKbQueue();
    ingestKbPath();
// Called by the socket layer when ingest completes or fails, so the next queue item can start.
}
// @param {string} status 'done' or 'failed'; shown as a badge in the queue list.


function markCurrentKbQueueDone(status) {
    const running = kbQueue.find(item => item.status === 'running');
    if (running) {
        running.status = status;
        renderKbQueue();
    }
}

// Validates the ingest form, then sends rpc id 51 (ingest_path). The reply/progress notifications
// are handled in karl_socket.js, which calls markCurrentKbQueueDone and ingestNextQueuedPath.
function ingestKbPath() {
    if (!isConnected()) {
        vscode.postMessage({ command: 'show_error', text: 'Karl is disconnected.' });
        return;
    }
    // Default chunk of 200 words and overlap 0 mirror RAGPipeline.ingest_file defaults on the Python side.
    const ingestPath = $('kbPath').value.trim();
    const chunkSize = Number($('kbChunkSize').value) || 200;
    const overlap = Number($('kbOverlap').value) || 0;
    if (!ingestPath) {
        vscode.postMessage({ command: 'show_error', text: 'Choose a file or folder to ingest.' });
        return;
    }
    // The sliding window would never advance if overlap >= chunk size, so reject it before the call.
    if (overlap >= chunkSize) {
        vscode.postMessage({ command: 'show_error', text: 'Overlap must be lower than chunk size.' });
        return;
    }
    // Disable the button until the reply arrives so the same ingest cannot be double-submitted.
    $('kbIngestBtn').disabled = true;
    $('kbIngestState').innerText = 'Starting';
    rpc(51, 'ingest_path', {
        path: ingestPath,
        recursive: $('kbRecursive').checked,
        chunk_size: chunkSize,
        overlap
    });
}

// Retrieval preview: runs the real RAG search (rpc id 52) so the user can tune top-k and the
// distance threshold before relying on them in chat.
function searchKb() {
    const query = $('kbQuery').value.trim();
    if (!query) {
        vscode.postMessage({ command: 'show_error', text: 'Enter a retrieval preview query.' });
        return;
    }
    // Immediate placeholder: search can take a moment while the embedder warms up.
    $('kbResults').innerHTML = '<div class="result-card">Searching index...</div>';
    // Recent queries are stored (up to 12) for the quick-pick list.
    rememberKbQuery(query);
    rpc(52, 'search_kb', {
        query,
        top_k: Number($('kbTopK').value) || 5,
        // 0 means "no distance cut-off".
        threshold: Number($('kbThreshold').value) || 0,
        // null = search all sources; a name restricts the search to one ingested file.
        source_filter: $('kbSourceFilter').value || null
    });
}

// Same RPC method as the knowledge-base search but with its own id (53) and result box, so the Swarm
// panel's reply is routed to renderSwarmRagResults instead of the KB panel.
function searchSwarmRag() {
    const query = $('swarmRagQuery').value.trim();
    if (!query) {
        vscode.postMessage({ command: 'show_error', text: 'Enter a search term.' });
        return;
    }
    $('swarmRagResults').innerHTML = '<div class="result-card">Searching index...</div>';
    rpc(53, 'search_kb', {
        query,
        top_k: Number($('swarmRagTopK').value) || 5,
        threshold: Number($('swarmRagThreshold').value) || 0
    });
}

// Renders the reply of rpc id 53. Each card offers Copy and "Inject Into Objective" buttons; the
// click handlers are delegated elsewhere using the data-idx attribute, which indexes lastSwarmRagResults.
function renderSwarmRagResults(payload) {
    // Normalise a null/odd payload so the function never throws inside the message handler.
    if (!payload || typeof payload !== 'object') payload = {};
    const results = Array.isArray(payload.results) ? payload.results : [];
    lastSwarmRagResults = results;
    $('swarmRagResults').innerHTML = results.length ? results.map((result, index) => {
        if (!result) result = {};
        return `
        <div class="result-card">
            <div class="result-meta">Rank ${escapeHtml(result.rank ?? index)} · ${escapeHtml(result.source_file || 'unknown')} · Chunk ${escapeHtml(result.chunk_id || '0')} · dist=${Number(result.distance || 0).toFixed(4)}</div>
            // Text is escaped (it is document content, untrusted) and cut to 1800 characters to keep the DOM small.
            <pre>${escapeHtml(result.text || '').slice(0, 1800)}</pre>
            <div class="action-row compact-actions" style="margin-top:6px;">
                <button data-idx="${index}" data-copy>Copy</button>
                <button data-idx="${index}" data-inject>Inject Into Objective</button>
            </div>
        </div>
    `; }).join('') : '<div class="result-card">No chunks matched the current query and threshold.</div>';
}

// Used for features the VS Code client offers in the UI but whose Python RPC does not exist yet
// (training validation, tokenizer preview...). Shows an explicit explanation rather than failing silently.
// @param {string} feature  human-readable feature name; also selects which log panel gets the note
// @param {string} method   the missing bridge method name
function requiresBridgeSupport(feature, method) {
    const text = `${feature} requires Karl bridge method "${method}". The VS Code workspace is ready, but this backend endpoint is not exposed by the current desktop bridge.`;
    vscode.postMessage({ command: 'show_error', text });
    // Echo the message into the panel belonging to that feature, so it is visible where the user clicked.
    if ($('trainingLog') && feature.toLowerCase().includes('training')) {
        $('trainingLog').innerText += `\n[Bridge Required] ${text}`;
    }
    if ($('evalLog') && feature.toLowerCase().includes('eval')) {
        $('evalLog').innerText = `[Bridge Required] ${text}`;
    }
    if ($('tokenPreview') && feature.toLowerCase().includes('tokenizer')) {
        $('tokenPreview').innerText = text;
    }
}

// Placeholder handler for the Training "Validate" button until the bridge exposes validation.
function validateTrainingForm() {
    vscode.postMessage({ command: 'show_error', text: 'Training validation requires bridge support.' });
}

// Remember a launched task for the "Recent tasks" quick list (newest first, same text+workflow
// de-duplicated, at most 15 kept) and persist it in the webview state.
function rememberTask(workflowId, title, objective, filepath) {
    // Whitespace-only objectives are not worth remembering.
    const text = String(objective || '').trim();
    if (!text) return;
    recentTasks = [
        { workflowId, title, objective: text, filepath: filepath || '', at: new Date().toISOString() },
        ...recentTasks.filter(item => item.objective !== text || item.workflowId !== workflowId)
    ].slice(0, 15);
    renderRecentTasks();
    persist();
}

// Same pattern for knowledge-base queries (limit 12).
function rememberKbQuery(query) {
    recentKbQueries = [query, ...recentKbQueries.filter(item => item !== query)].slice(0, 12);
    renderRecentKbQueries();
    persist();
}

// Adds a running entry to the task queue panel and returns its id so the caller can finish it later.
function addTask(mode, objective) {
    // Millisecond timestamp ids are unique enough because tasks are started by a human click.
    const id = `task-${Date.now()}`;
    taskQueue.unshift({ id, mode, objective, status: 'running', at: new Date().toISOString() });
    renderTaskQueue();
    return id;
}

// Marks the active task done/failed and clears the active id. map() makes a new array so the render
// function sees an updated reference.
function finishActiveTask(status) {
    // Nothing to do when no task is active (e.g. a stop arrived twice).
    if (!activeTaskId) return;
    taskQueue = taskQueue.map(task => task.id === activeTaskId ? { ...task, status } : task);
    activeTaskId = '';
    renderTaskQueue();
}

// Client-side filter of the codex topic list; no RPC. Matches case-insensitively on item text
// and toggles `display` (grid because .source-item uses a grid layout).
function filterCodex() {
    const q = $('codexSearch').value.toLowerCase();
    $('codexList').querySelectorAll('.source-item').forEach(item => {
        item.style.display = item.innerText.toLowerCase().includes(q) ? 'grid' : 'none';
    });
}

// Collects the generation settings used by every chat/lab/swarm request. `|| default` handles
// empty or NaN inputs; the defaults (0.7/0.95/2048/5/0) match the Python workbench defaults.
function hyperparams() {
    return {
        temperature: Number($('karlTemp').value) || 0.7,
        top_p: Number($('karlTopP').value) || 0.95,
        max_tokens: Number($('karlMaxTok').value) || 2048,
        rag_enabled: $('karlRag').checked,
        agentic_loop_enabled: $('karlLoop').checked,
        rag_top_k: Number($('kbTopK').value) || 5,
        rag_threshold: Number($('kbThreshold').value) || 0
    };
}

// ── WS-state MutationObserver ─────────────────────────────────────────────────
// ── WS-state MutationObserver ─────────────────────────────────────────────────
// Watches #statusDot className changes and mirrors the state to
// body.dataset.wsState so CSS can drive the chat-border animation.
// Detail: a MutationObserver reacts to class changes on the status dot. While a reply is generating,
// the "generating" state is left alone so the animation is not interrupted.
function _initWsStateObserver() {
    // The status dot class is the single source of truth for connection state.
    const dot = $('statusDot');
    if (!dot) return;
    // Take the first class that is not the base class, e.g. "connected" or "offline".
    function _sync() {
        const state = [...dot.classList].find(c => c !== 'status-dot') || 'offline';
        // Do not overwrite "generating": sendChatMessage sets it and handleChatFinished clears it.
        if (document.body.dataset.wsState !== 'generating') {
            document.body.dataset.wsState = state;
        }
    }
    // Initial sync, since the observer only reports later changes.
    _sync();
    // attributeFilter limits callbacks to class changes, so the observer does not fire for unrelated edits.
    new MutationObserver(_sync).observe(dot, { attributes: true, attributeFilter: ['class'] });
}

// ── Chat workspace inner tabs ─────────────────────────────────────────────────
// Injects a 3-tab sub-layout (Chat / Reasoning / Swarm) inside #workspace-chat.
// Runs after _initThoughtsPanel() so #introspectionBox already has its header.
// Detail: the Chat tab is restructured at run time (existing elements are moved into new panels), guarded by
// a data attribute so it only happens once.
function _initChatInnerTabs() {
    const chatSection = $('workspace-chat');
    // Idempotency guard: a second call (hot reload, repeated init) would otherwise nest the panels again.
    if (!chatSection || chatSection.dataset.innerTabsReady) return;
    // Guard: never restructure the tab twice.
    chatSection.dataset.innerTabsReady = '1';

    // Existing nodes are looked up here and re-parented below, preserving their ids so every other
    // function that calls $('chatMessages') etc. keeps working.
    const thoughtsBox  = $('introspectionBox');
    const chatMessages = $('chatMessages');
    const composer     = chatSection.querySelector('.composer');
    const sectionHead  = chatSection.querySelector('.section-head');
    // Bail out if the markup changed and an expected element is missing, rather than half-building the tabs.
    if (!thoughtsBox || !chatMessages || !composer) return;

    // ── Tab bar ────────────────────────────────────────────────────────────────
    // ── Tab bar ────────────────────────────────────────────────────────────────
    const tabBar = document.createElement('div');
    tabBar.className = 'chat-inner-tabs';
    // data-inner-tab names the panel id to show; the click handler for these buttons is wired below.
    tabBar.innerHTML = `
        <button class="chat-inner-tab active" data-inner-tab="chatPanel">Chat</button>
        <button class="chat-inner-tab" data-inner-tab="reasoningPanel">Reasoning</button>
        <button class="chat-inner-tab" data-inner-tab="swarmPanel">Swarm</button>
        // Thoughts toggle lives in the bar so it stays visible whichever panel is open.
        <button class="chat-toggle active" id="thoughtsVisToggle" title="Show / hide inline thought stream">◈ Thoughts</button>
    `;

    // appendChild moves existing nodes (it does not copy them), preserving their event listeners.
    // ── Panel 1: chat messages + composer ─────────────────────────────────────
    const chatPanel = document.createElement('div');
    chatPanel.id = 'chatPanel';
    chatPanel.className = 'chat-tab-panel active';
    chatPanel.appendChild(thoughtsBox);
    chatPanel.appendChild(chatMessages);
    chatPanel.appendChild(composer);

    // ── Panel 2: full reasoning view ──────────────────────────────────────────
    // ── Panel 2: full reasoning view ──────────────────────────────────────────
    const reasoningPanel = document.createElement('div');
    reasoningPanel.id = 'reasoningPanel';
    reasoningPanel.className = 'chat-tab-panel';
    // The #reasoningFullView <pre> is filled by the thought-token handler in karl_render.js.
    reasoningPanel.innerHTML = `
        <div style="margin-bottom:8px;">
            <div class="eyebrow">Full Reasoning Stream</div>
            <div style="color:var(--karl-muted);font-size:10px;margin-top:3px;">Complete thought stream — updated live during generation</div>
        </div>
        <pre id="reasoningFullView" class="reasoning-full">No reasoning yet. Start a chat to see thoughts here.</pre>
    `;

    // ── Panel 3: swarm progress feed ──────────────────────────────────────────
    // ── Panel 3: swarm progress feed ──────────────────────────────────────────
    const swarmPanel = document.createElement('div');
    swarmPanel.id = 'swarmPanel';
    swarmPanel.className = 'chat-tab-panel';
    swarmPanel.innerHTML = `
        <div style="margin-bottom:8px;">
            <div class="eyebrow">Swarm Progress Feed</div>
            <div style="color:var(--karl-muted);font-size:10px;margin-top:3px;">Real-time agent execution timeline</div>
        </div>
        // #swarmFeed receives timeline cards from the swarm notification handlers; data-swarm-empty marks
        // the placeholder that is removed on the first event.
        <div id="swarmFeed" class="swarm-feed">
            <div style="color:var(--karl-muted);padding:8px;font-size:11px;" data-swarm-empty>No swarm activity yet.</div>
        </div>
    `;

    // Insert all panels right after the section-head
    // Insert the new panels right under the section header; without one, fall back to prepending them.
    // after() inserts siblings in order, so the final DOM order is: bar, chat, reasoning, swarm.
    if (sectionHead) {
        sectionHead.after(tabBar, chatPanel, reasoningPanel, swarmPanel);
    } else {
        // prepend() puts its arguments in the given order, hence the reversed list to get the same final order.
        chatSection.prepend(swarmPanel, reasoningPanel, chatPanel, tabBar);
    }

    // ── Tab switching ──────────────────────────────────────────────────────────
    // One delegated click handler per tab button; _switchChatInnerTab owns the show/hide logic.
    tabBar.querySelectorAll('.chat-inner-tab').forEach(btn => {
        btn.addEventListener('click', () => _switchChatInnerTab(btn.dataset.innerTab));
    });

    // ── Thoughts inline visibility toggle ─────────────────────────────────────
    // Button that hides/shows the inline reasoning box in the Chat panel.
    // Only the inline box is hidden: the same text stays available in the Reasoning panel.
    const toggleBtn = $('thoughtsVisToggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            // An empty style.display means "visible by stylesheet", so '' is the visible state and 'none' hidden.
            const hidden = thoughtsBox.style.display === 'none';
            thoughtsBox.style.display = hidden ? '' : 'none';
            // toggle(name, force): the button looks active when the thoughts are visible.
            toggleBtn.classList.toggle('active', hidden);
        });
    }
}

/**
 * Show one of the chat sub-panels (chatPanel, reasoningPanel, swarmPanel) and mark its tab button active.
 * @param {string} panelId
 */
function _switchChatInnerTab(panelId) {
    // The ids of the three panels created by _initChatInnerTabs; only one carries .active (CSS shows it).
    const panels = ['chatPanel', 'reasoningPanel', 'swarmPanel'];
    panels.forEach(id => {
        const el = $(id);
        // classList.toggle with a boolean adds or removes the class in one call, without needing an if/else.
        if (el) el.classList.toggle('active', el.id === panelId);
    });
    const chatSection = $('workspace-chat');
    if (chatSection) {
        // Keep the tab-button highlight in sync with the visible panel.
        chatSection.querySelectorAll('.chat-inner-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.innerTab === panelId);
        });
    }
}

// ── Logs workspace subtabs ────────────────────────────────────────────────────
// Adds three subtabs (Swarm / Training / Eval) inside #workspace-logs,
// wrapping the existing #terminal, #trainingLog, #evalLog into panels.
// Idempotent like _initChatInnerTabs: the data attribute marks the logs area as already restructured.
// Detail: existing log elements are moved into panels, guarded by a data attribute so it only happens once.
function _initLogsSubtabs() {
    const logsSection = $('workspace-logs');
    if (!logsSection || logsSection.dataset.logsSubtabsReady) return;
    // Missing log element means the HTML changed; skip rather than build a partial UI.
    logsSection.dataset.logsSubtabsReady = '1';

    const terminal    = $('terminal');
    const trainingLog = $('trainingLog');
    const evalLog     = $('evalLog');
    // Remember the surrounding rows so the new sub-tabs can be inserted at the right spot.
    const actionRow   = logsSection.querySelector('.action-row');
    const sectionHead = logsSection.querySelector('.section-head');
    if (!terminal || !trainingLog || !evalLog) return;

    // Remove orphaned eyebrow labels that belonged to trainingLog / evalLog
    // Old static labels would now be duplicated by the sub-tab buttons, so remove them.
    logsSection.querySelectorAll(':scope > .eyebrow').forEach(el => el.remove());

    // ── Subtab bar ────────────────────────────────────────────────────────────
    // ── Subtab bar ────────────────────────────────────────────────────────────
    // data-log-panel holds the id of the panel that button reveals (used in the click handler below).
    const subtabBar = document.createElement('div');
    subtabBar.className = 'log-subtabs';
    subtabBar.innerHTML = `
        <button class="log-subtab active" data-log-panel="swarmLogPanel">Swarm</button>
        <button class="log-subtab"        data-log-panel="trainingLogPanel">Training</button>
        <button class="log-subtab"        data-log-panel="evalLogPanel">Eval</button>
    `;

    // Wrapping the existing <pre>/<div> elements (rather than cloning them) keeps ids and any code
    // that appends to $('terminal'), $('trainingLog') or $('evalLog') working unchanged.
    // ── Panels ────────────────────────────────────────────────────────────────
    const swarmLogPanel = document.createElement('div');
    swarmLogPanel.id = 'swarmLogPanel';
    swarmLogPanel.className = 'log-panel active';
    swarmLogPanel.appendChild(terminal);

    const trainingLogPanel = document.createElement('div');
    trainingLogPanel.id = 'trainingLogPanel';
    trainingLogPanel.className = 'log-panel';
    trainingLogPanel.appendChild(trainingLog);

    const evalLogPanel = document.createElement('div');
    evalLogPanel.id = 'evalLogPanel';
    evalLogPanel.className = 'log-panel';
    evalLogPanel.appendChild(evalLog);

    // Insert after action-row (or section-head as fallback)
    // Place the sub-tabs below the action row, or below the header if there is no action row.
    const insertRef = actionRow || sectionHead;
    // Without a reference node there is nowhere sensible to place the sub-tabs, so leave the logs as-is.
    if (insertRef) {
        insertRef.after(subtabBar, swarmLogPanel, trainingLogPanel, evalLogPanel);
    }

    // ── Tab switching ─────────────────────────────────────────────────────────
    // Click handler: clear all active tab buttons, activate the clicked one, then show only its panel.
    subtabBar.querySelectorAll('.log-subtab').forEach(btn => {
        btn.addEventListener('click', () => {
            subtabBar.querySelectorAll('.log-subtab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ['swarmLogPanel', 'trainingLogPanel', 'evalLogPanel'].forEach(id => {
                const el = $(id);
                if (el) el.classList.toggle('active', el.id === btn.dataset.logPanel);
            });
        });
    });
}

// ── Educational Sandbox Workspace ─────────────────────────────────────────────
// ── Educational Sandbox Workspace ─────────────────────────────────────────────
// A teaching feature: lets a learner see TF-IDF and a (simulated) Mini-GPT training run without
// leaving the editor. Real training is requested over RPC; the animation is purely visual.
// Interval handle for the fake training animation; kept so a second click can cancel the first run.
// State for the educational Mini-GPT / TF-IDF simulations further below.
let miniGptSimTimer = null;
// Step counter used when the trainer prints loss without a step number.
let currentMiniStep = 0;

/**
 * Build a TF-IDF model for a few short documents (teaching demo; the same maths as scikit-learn's defaults).
 *
 * Steps: lower-case words (2+ chars, minus common stop words); idf = ln((1+N)/(1+df)) + 1 (smoothed);
 * weight = term count x idf; each document vector is scaled to unit length (L2 norm).
 * @param {string[]} documents
 * @returns {{vocabulary: {word: string, df: number, idf: number}[], vectors: Object<string, number>[]}}
 */
function fitTfidfVectorizer(documents) {
    // Very small stop-word list; only meant to keep the demo output readable.
    // Deliberately tiny stop-word list: enough to hide the most common function words in a demo, not a
    // replacement for scikit-learn's list.
    const stopWords = new Set(["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "at", "for", "with", "by", "about", "as"]);
    
    // Tokeniser: lowercase, split on every non-alphanumeric run, drop one-letter words and stop words.
    function tokenize(text) {
        return text.toLowerCase()
            .split(/[^a-z0-9]+/)
            .filter(word => word.length >= 2 && !stopWords.has(word));
    }
    
    // Tokenise once; the rest of the function works on these arrays.
    const tokenizedDocs = documents.map(doc => tokenize(doc));
    // termDocCounts[word] = df, the number of documents containing the word (not the total count).
    const termDocCounts = {};
    
    tokenizedDocs.forEach(tokens => {
        // A Set counts each word once per document, which is exactly the definition of document frequency.
        const uniqueTokens = new Set(tokens);
        uniqueTokens.forEach(word => {
            termDocCounts[word] = (termDocCounts[word] || 0) + 1;
        });
    });
    
    // N = number of documents; the idf formula below needs it.
    const N = documents.length;
    const vocabulary = [];
    
    for (const [word, df] of Object.entries(termDocCounts)) {
        // Smoothing (+1) avoids division by zero and keeps idf positive for terms found in every document.
        const idf = Math.log((1 + N) / (1 + df)) + 1.0;
        vocabulary.push({ word, df, idf });
    }
    
    // Alphabetical order gives a stable table that does not change between runs.
    vocabulary.sort((a, b) => a.word.localeCompare(b.word));
    
    // One vector per document, keyed by vocabulary word, so every vector has the same dimensions.
    const vectors = tokenizedDocs.map((tokens, docIdx) => {
        // tf = raw term counts in this document.
        const tf = {};
        tokens.forEach(word => {
            tf[word] = (tf[word] || 0) + 1;
        });
        
        // vector holds count * idf before normalisation; normSq accumulates the squared length.
        const vector = {};
        let normSq = 0;
        vocabulary.forEach(v => {
            const word = v.word;
            const count = tf[word] || 0;
            const tfidf = count * v.idf;
            vector[word] = tfidf;
            normSq += tfidf * tfidf;
        });
        
        // L2 normalisation so that long documents do not dominate cosine similarity.
        // L2 normalisation so that long documents do not dominate cosine similarity.
        const norm = Math.sqrt(normSq);
        const normVector = {};
        vocabulary.forEach(v => {
            const word = v.word;
            // Guard against an all-zero vector (a document made only of stop words), which would divide by zero.
            normVector[word] = norm > 0 ? (vector[word] / norm) : 0;
        });
        
        return normVector;
    });
    
    return { vocabulary, vectors };
}

/**
 * Show the vocabulary table (word, document frequency, idf) and each document's non-zero TF-IDF terms.
 * @param {{word: string, df: number, idf: number}[]} vocabulary
 * @param {Object<string, number>[]} vectors
 * @param {string[]} docs
 */
function renderVectorizerResults(vocabulary, vectors, docs) {
    // Rebuild the table body from scratch on every fit.
    const vocabBody = $('vocabTable').querySelector('tbody');
    vocabBody.innerHTML = '';
    
    vocabulary.forEach(v => {
        const row = document.createElement('tr');
        // Cells are right-aligned numbers except the word; the word is escaped because documents are user input.
        row.innerHTML = `
            <td style="text-align: left; padding: 4px; color: var(--karl-text);">${escapeHtml(v.word)}</td>
            <td style="text-align: right; padding: 4px; color: var(--karl-muted);">${v.df}</td>
            <td style="text-align: right; padding: 4px; color: var(--karl-accent-2);">${v.idf.toFixed(4)}</td>
        `;
        vocabBody.appendChild(row);
    });
    
    // Build a plain-text listing: only non-zero terms are shown, since most entries of a sparse TF-IDF
    // vector are zero and would drown the useful numbers.
    let vectorText = '';
    vectors.forEach((v, idx) => {
        vectorText += `Doc ${idx + 1}: "${escapeHtml(docs[idx])}"\n`;
        const nonZeroTerms = [];
        vocabulary.forEach(term => {
            const val = v[term.word];
            if (val > 0) {
                nonZeroTerms.push(`${term.word}: ${val.toFixed(4)}`);
            }
        });
        vectorText += `  Vector: [${nonZeroTerms.join(', ')}]\n\n`;
    });
    
    // innerText (not innerHTML) is used because the text already contains escaped markup characters.
    $('tfidfVectors').innerText = vectorText;
    // The output panel is hidden until the first successful fit.
    $('vectorizerOutput').style.display = 'block';
}

/**
 * Parse a backend log line from the Mini-GPT trainer and update the sandbox training display.
 *
 * Recognises "Step N | Loss: x" lines and JSON-style "loss": x values (the latter increments a local step counter),
 * replaces the sample output panel for "Generation Output" lines, and appends other training messages to the log.
 * @param {string} msg
 */
function handleInterceptedTelemetry(msg) {
    // Telemetry may arrive as non-strings (null params); ignore those.
    if (typeof msg !== 'string') return;
    
    // The status panel exists only in the educational layout; do nothing when it is absent.
    const statusPanel = $('miniGptTrainingStatus');
    if (!statusPanel) return;
    statusPanel.style.display = 'block';
    
    // Matches the trainer's log format, e.g. "Step 50 | Val Loss: 2.1234".
    // Matches the trainer's log format, e.g. "Step 50 | Val Loss: 2.1234".
    const stepLossRegex = /Step\s*(\d+)\s*\|\s*(?:Val\s+)?Loss:\s*([0-9.]+)/i;
    const match = msg.match(stepLossRegex);
    if (match) {
        const step = parseInt(match[1]);
        const loss = parseFloat(match[2]);
        updateMiniGptTelemetryUi(step, loss);
        return;
    }
    
    // Fallback for metrics printed as a dict/JSON fragment such as {'loss': 1.23}.
    // Fallback for metrics printed as a dict/JSON fragment such as {'loss': 1.23}.
    const jsonLossRegex = /['"]loss['"]\s*:\s*([0-9.]+)/i;
    const jsonMatch = msg.match(jsonLossRegex);
    if (jsonMatch) {
        const loss = parseFloat(jsonMatch[1]);
        // No step in the line, so count lines ourselves; currentMiniStep resets only on page reload.
        currentMiniStep++;
        updateMiniGptTelemetryUi(currentMiniStep, loss);
        return;
    }
    
    // Sample text from the model. HTML tags are stripped and the rest is escaped before display.
    // Sample text from the model. HTML tags are stripped and the rest is escaped before display.
    if (msg.includes('Generation Output') || msg.includes('--- Generation')) {
        const cleanMsg = msg.replace(/<[^>]*>/g, '');
        $('miniTypewriterOutput').innerHTML = `<div style="color: var(--karl-accent);">${escapeHtml(cleanMsg)}</div>`;
        return;
    }
    
    // Keyword filter so unrelated status_update messages (model loaded, etc.) do not flood the loss log.
    if (msg.includes('training') || msg.includes('Trainer') || msg.includes('Loss') || msg.includes('Epoch')) {
        const cleanMsg = msg.replace(/<[^>]*>/g, '');
        const entry = document.createElement('div');
        entry.style.borderBottom = '1px solid rgba(0, 194, 255, 0.05)';
        entry.style.padding = '2px 0';
        entry.innerText = cleanMsg;
        const logContainer = $('miniLossHistory');
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

/**
 * Run a purely local, fake training animation (loss falls from about 4.5 towards 0.12 with noise, random words as "samples").
 *
 * It is a teaching visual that runs alongside the real ``start_mini_train`` request; it is NOT real training data.
 * It advances 5 steps every 500 ms.
 * @param {number} maxIters
 * @param {number} lr
 */
function startMiniGptSimulation(maxIters, lr) {
    // Cancel a still-running animation before starting a new one, otherwise two timers would race.
    if (miniGptSimTimer) {
        clearInterval(miniGptSimTimer);
    }
    
    // Make the sandbox panel visible and clear the previous run's log and sample text.
    $('miniGptTrainingStatus').style.display = 'block';
    $('miniLossHistory').innerHTML = '<div style="color: var(--karl-muted);">Local Simulation Active...</div>';
    $('miniTypewriterOutput').innerHTML = '';
    
    let step = 0;
    // Start loss near 4.5 (about ln(vocab) for a fresh model) plus noise, like a real from-scratch run.
    let currentLoss = 4.5 + Math.random();
    
    // Words used to compose the fake "generated" sentences; chosen to match Karl's theme.
    const sampleWords = [
        "karl", "agent", "local", "model", "training", "offline", "zero", "network", 
        "vector", "sandbox", "embeddings", "attention", "transformer", "weights", 
        "loss", "gradient", "optimization", "completion", "intelligence", "generation"
    ];
    
    // 10-24 random words and a full stop: enough to look like a sentence.
    function generateTypewriterTokens() {
        const wordCount = 10 + Math.floor(Math.random() * 15);
        const sentence = [];
        for (let i = 0; i < wordCount; i++) {
            sentence.push(sampleWords[Math.floor(Math.random() * sampleWords.length)]);
        }
        return sentence.join(' ') + '.';
    }
    
    // 500 ms tick advancing 5 steps: a 100-iteration demo lasts about 10 seconds.
    miniGptSimTimer = setInterval(() => {
        step += 5;
        // Past the requested iteration count: stop the timer and print a completion line.
        if (step > maxIters) {
            clearInterval(miniGptSimTimer);
            miniGptSimTimer = null;
            const finalEntry = document.createElement('div');
            finalEntry.style.color = 'var(--karl-good)';
            // Message is informational only; no file is actually written by the simulation.
            finalEntry.innerText = `[Simulation] Training complete. Saved to data/mini_gpt/weights.pt`;
            $('miniLossHistory').appendChild(finalEntry);
            return;
        }
        
        // Loss drifts down linearly by up to 3.0 over the run; the random term adds +/-0.2 jitter; never below 0.12.
        // Loss drifts down linearly by up to 3.0 over the run; the random term adds +/-0.2 jitter; never below 0.12.
        const decay = (step / maxIters) * 3.0;
        currentLoss = Math.max(0.12, (4.5 - decay) + (Math.random() * 0.4 - 0.2));
        
        updateMiniGptTelemetryUi(step, currentLoss);
        
        // Every 20 steps, show a new random "generated" sentence.
        // Every 20 steps, show a new random "generated" sentence.
        if (step % 20 === 0) {
            const text = generateTypewriterTokens();
            streamTypewriterTokens(step, maxIters, text);
        }
    }, 500);
}

/**
 * Show the current step/loss and append a log line (green once the loss drops below 1.0).
 * @param {number} step
 * @param {number} loss
 */
function updateMiniGptTelemetryUi(step, loss) {
    // Either element may be missing in a reduced layout; update whichever exists.
    const stepEl = $('miniCurrentStep');
    const lossEl = $('miniCurrentLoss');
    if (stepEl) stepEl.innerText = step;
    if (lossEl) lossEl.innerText = loss.toFixed(4);
    
    const entry = document.createElement('div');
    // Green once the loss falls below 1.0, a rough "now learning" signal for beginners.
    entry.style.color = loss < 1.0 ? 'var(--karl-good)' : 'var(--karl-text)';
    entry.innerText = `Step ${step} | Loss: ${loss.toFixed(4)}`;
    
    // Auto-scroll so the newest line is visible without manual scrolling.
    const logContainer = $('miniLossHistory');
    if (logContainer) {
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

/**
 * Show ``text`` one character at a time (every 20 ms) under a "Step N / M" heading, like a model typing.
 * @param {number} step
 * @param {number} maxIters
 * @param {string} text
 */
function streamTypewriterTokens(step, maxIters, text) {
    // Missing container means the educational panel is not in this layout.
    const container = $('miniTypewriterOutput');
    if (!container) return;
    
    // Heading line resets the area, so each sample replaces the previous one.
    container.innerHTML = `<div style="color: var(--karl-accent); border-bottom: 1px solid rgba(0, 194, 255, 0.1); margin-bottom: 4px; padding-bottom: 2px;">--- Step ${step} / ${maxIters} ---</div>`;
    
    // Append characters to a span rather than rewriting innerHTML, which would restart the whole layout.
    const textSpan = document.createElement('span');
    container.appendChild(textSpan);
    
    let charIdx = 0;
    // 20 ms per character (50 chars/s) imitates streaming tokens; the interval stops itself at the end.
    const interval = setInterval(() => {
        if (charIdx >= text.length) {
            clearInterval(interval);
            return;
        }
        textSpan.innerText += text[charIdx];
        charIdx++;
        container.scrollTop = container.scrollHeight;
    }, 20);
}

/**
 * Wire the two sandbox buttons and hook the socket message handler.
 *
 * "Fit vectorizer" sends the documents to the backend (rpc id 60) and also computes TF-IDF locally for instant display.
 * "Start training" sends ``start_mini_train`` (rpc id 61) and starts the local animation.
 */
function _initEducationalSandbox() {
    // Lookup is guarded: the buttons exist only when the sandbox panel is present in the HTML.
    const fitBtn = $('fitVectorizerBtn');
    const trainBtn = $('startMiniGptBtn');
    
    // Fit button: validate input, tell the server, and also compute locally so the user sees results
    // instantly even when the bridge is slow or the method is unavailable.
    if (fitBtn) {
        fitBtn.addEventListener('click', () => {
            const text = $('sandboxDocs').value.trim();
            if (!text) {
                vscode.postMessage({ command: 'show_error', text: 'Please enter at least one document.' });
                return;
            }
            // One document per non-empty line.
            const docs = text.split('\n').map(d => d.trim()).filter(d => d.length > 0);
            if (docs.length === 0) {
                vscode.postMessage({ command: 'show_error', text: 'Please enter valid non-empty documents.' });
                return;
            }
            
            // rpc id 60 -> fit_vectorizer (server-side scikit-learn); its reply is shown by karl_socket.js.
            rpc(60, 'fit_vectorizer', { documents: docs });
            
            // Local computation is wrapped in try/catch so an exception in the demo cannot break the page.
            try {
                const { vocabulary, vectors } = fitTfidfVectorizer(docs);
                renderVectorizerResults(vocabulary, vectors, docs);
            } catch (err) {
                log(`[Vectorizer Local Error] ${err.message}`);
            }
        });
    }

    // Train button: reads hyperparameters with defaults (lr 0.001, 100 iterations, batch 16).
    if (trainBtn) {
        trainBtn.addEventListener('click', () => {
            const lr = parseFloat($('miniLr').value) || 0.001;
            const iters = parseInt($('miniIters').value) || 100;
            const batchSize = parseInt($('miniBatchSize').value) || 16;
            
            // rpc id 61 -> start_mini_train launches the real trainer on the backend; log lines come back as
            // auto_train_log / status_update notifications handled by handleInterceptedTelemetry.
            rpc(61, 'start_mini_train', {
                lr: lr,
                max_iters: iters,
                batch_size: batchSize
            });
            
            // The local animation runs immediately for feedback, even before the server answers.
            startMiniGptSimulation(iters, lr);
        });
    }

    // Only patch if karl_socket.js defined the handler; this keeps karl.js loadable on its own.
    if (typeof handleSocketMessage === 'function') {
        // Wrap handleSocketMessage (monkey-patch): run the original handler, then also feed trainer log lines to the
        // sandbox display. This avoids editing karl_socket.js. It relies on the function being reassignable
        // (a plain, non-module script).
        const originalHandleSocketMessage = handleSocketMessage;
        handleSocketMessage = function(data) {
            originalHandleSocketMessage(data);
            
            // auto_train_log carries trainer stdout lines; status_update carries state messages. Both go through
            // the same parser, which ignores lines it does not recognise.
            if (data && data.method === 'auto_train_log') {
                const msg = (data.params && data.params.message) || '';
                handleInterceptedTelemetry(msg);
            } else if (data && data.method === 'status_update') {
                const msg = (data.params && data.params.message) || '';
                handleInterceptedTelemetry(msg);
            }
        };
    }
}

// Second DOMContentLoaded handler (the first is at the top of this file); it runs after it, so the
// elements and handleSocketMessage already exist.
window.addEventListener('DOMContentLoaded', () => {
    _initEducationalSandbox();
});

/**
 * Renders an Obsidian-themed crash diagnostic callout in the chat panel when the
 * llama-cpp-python subprocess exits with SIGSEGV (139) or OOM SIGKILL (137).
 * @param {{ reason?: string, message?: string, exitCode?: number|null, exitSignal?: string|null }} msg
 */
function _renderCompletionDiagnostic(msg) {
    // Pick colours/text per crash type: SIGSEGV (warning, usually a CPU without AVX) or OOM kill (danger).
    // Pick colours/text per crash type: SIGSEGV (warning, usually a CPU without AVX) or OOM kill (danger).
    const chatEl = $('chatMessages');
    if (!chatEl) return;

    // reason values are produced by the extension host when the completion subprocess dies (exit code 139 or 137).
    const isSegv = msg.reason === 'sigsegv';
    const isOom  = msg.reason === 'oom';

    // CSS custom properties (--karl-warn / --karl-danger) come from the theme, so the card follows light/dark.
    const accentVar   = isSegv ? 'var(--karl-warn)' : 'var(--karl-danger)';
    // Inline SVG icons (no external assets, because the webview CSP forbids remote content).
    const iconSvg     = isSegv
        ? `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:-2px"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 3.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 4.5zm0 7a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75z"/></svg>`
        : `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:-2px"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm3.53 4.47a.75.75 0 0 1 0 1.06L9.06 9l2.47 2.47a.75.75 0 0 1-1.06 1.06L8 10.06l-2.47 2.47a.75.75 0 0 1-1.06-1.06L6.94 9 4.47 6.53a.75.75 0 0 1 1.06-1.06L8 7.94l2.47-2.47a.75.75 0 0 1 1.06 0z"/></svg>`;
    const calloutType = isSegv ? 'warning' : 'danger';
    const title       = isSegv ? 'AVX Segmentation Fault' : 'Subprocess OOM Kill';
    const bodyText    = isSegv
        ? 'The local completion binary crashed with <strong>SIGSEGV</strong>. This usually means your CPU does not support the AVX/AVX2 instruction sets the binary was compiled for.'
        : 'The local completion subprocess was <strong>killed by the OS</strong> due to insufficient memory. Try loading a smaller model or reducing context size.';

    // Only the SIGSEGV case shows the rebuild command (CPU-compatible build of llama-cpp-python).
    const rebuildBlock = isSegv ? `
<div style="margin-top:.6rem">
  <div style="font-size:.7rem;opacity:.55;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem">Rebuild command</div>
  <pre style="margin:0;padding:.4rem .55rem;background:rgba(0,0,0,.35);border-radius:4px;font-size:.75rem;color:var(--karl-good);overflow-x:auto;white-space:pre-wrap;word-break:break-all">CMAKE_ARGS="-DGGML_AVX=OFF -DGGML_AVX2=OFF" pip install --force-reinstall --no-cache-dir llama-cpp-python</pre>
</div>` : '';

    // Show raw exit details (code/signal) when known; '—' stands for "unknown".
    const exitMeta = (msg.exitCode != null || msg.exitSignal)
        ? `<div style="margin-top:.45rem;font-size:.7rem;opacity:.45;font-family:monospace">exit&nbsp;${msg.exitCode ?? '—'} &nbsp;·&nbsp; signal&nbsp;${escapeHtml(msg.exitSignal || '—')}</div>`
        : '';

    // All dynamic text is escaped; bodyText/rebuildBlock are fixed strings defined above, not user data.
    const card = document.createElement('div');
    card.className = 'message assistant';
    card.dataset.diagnosticType = calloutType;
    // color-mix() tints the card background and border from the accent colour so one template serves both types.
    card.innerHTML = `
<div style="
  background:color-mix(in srgb,${accentVar} 8%,var(--karl-panel,#1e1e2e) 92%);
  border:1px solid color-mix(in srgb,${accentVar} 45%,transparent 55%);
  border-left:3px solid ${accentVar};
  border-radius:6px;
  padding:.55rem .75rem .65rem;
  margin:.2rem 0;
  box-shadow:0 0 12px color-mix(in srgb,${accentVar} 18%,transparent 82%);
">
  <div style="display:flex;align-items:center;gap:.35rem;color:${accentVar};font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.35rem">
    ${iconSvg}&nbsp;${escapeHtml(title)}
  </div>
  <div style="font-size:.8rem;line-height:1.5;opacity:.88">${bodyText}</div>
  ${rebuildBlock}
  ${exitMeta}
</div>`;
    // Append and scroll to the bottom so the diagnostic is visible immediately.
    chatEl.appendChild(card);
    chatEl.scrollTop = chatEl.scrollHeight;
}
