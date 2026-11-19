// ==============================================================================
// Program: karl_render.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Visual rendering engine formatting markdown, syntax-highlighted code blocks, and thoughts.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// ── module-level streaming state ───────────────────────────────────────────────
// Streaming state owned by this file: an approximate count of reasoning tokens shown so far, and the
// chat bubble that currently holds the blinking cursor. The leading underscore marks them as private.
let _thoughtTokenCount = 0;
let _streamingTarget = null;   // the .message-content el that owns the cursor

// ── helpers ────────────────────────────────────────────────────────────────────

/**
 * Escape the five HTML-special characters so untrusted text (model output, file names, errors)
 * can be placed inside innerHTML safely. null/undefined become an empty string.
 * @param {any} value
 * @returns {string}
 */
function escapeHtml(value) {
    // Step 1: Initialize function scope and validate execution context
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

// Elements with a scroll already queued; prevents scheduling many scrolls per frame while tokens stream in.
const _pendingScrolls = new Set();
/**
 * Scroll ``el`` to the bottom on the next animation frame (at most one pending scroll per element).
 * @param {HTMLElement | null} el
 */
function _triggerScroll(el) {
    // Step 1: Initialize function scope and validate execution context
    if (!el || _pendingScrolls.has(el)) return;
    _pendingScrolls.add(el);
    requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
        _pendingScrolls.delete(el);
    });
}

// ── streaming cursor ───────────────────────────────────────────────────────────

/**
 * Return the single shared streaming-cursor element, creating it if needed (hidden from screen readers).
 */
function _getOrCreateCursor() {
    // Step 1: Initialize function scope and validate execution context
    let cursor = document.getElementById('streaming-cursor');
    if (!cursor) {
        cursor = document.createElement('span');
        cursor.id = 'streaming-cursor';
        cursor.className = 'streaming-cursor';
        cursor.setAttribute('aria-hidden', 'true');
    }
    return cursor;
}

/**
 * Move the blinking cursor to the end of ``contentEl`` and make it the active streaming target.
 * @param {HTMLElement} contentEl
 */
function attachStreamingCursor(contentEl) {
    // Step 1: Initialize function scope and validate execution context
    removeStreamingCursor();
    _streamingTarget = contentEl;
    contentEl.appendChild(_getOrCreateCursor());
}

/**
 * Remove the cursor from the page and forget the streaming target (generation finished).
 */
function removeStreamingCursor() {
    // Step 1: Initialize function scope and validate execution context
    const cursor = document.getElementById('streaming-cursor');
    if (cursor) cursor.remove();
    _streamingTarget = null;
}

// Append plain text to a chat bubble's content element, keeping the cursor at
// the very end of the text at all times so it appears to advance with output.
// Each batch is wrapped in .token-appear for the CSS entrance animation.
// Detail: the cursor is removed before adding text and re-added afterwards, so it always stays last.
// Consecutive text reuses one span instead of creating a node per token (cheaper DOM updates).
function _appendChatText(el, text) {
    // Step 1: Initialize function scope and validate execution context
    if (!text) return;
    const cursor = document.getElementById('streaming-cursor');
    const cursorOwned = cursor && cursor.parentElement === el;
    if (cursorOwned) cursor.remove();
    
    // textContent (not innerHTML) is used for tokens, so streamed text can never inject markup.
    const lastChild = el.lastElementChild;
    if (lastChild && lastChild.className === 'token-appear') {
        lastChild.textContent += text;
    } else {
        const span = document.createElement('span');
        span.className = 'token-appear';
        span.textContent = text;
        el.appendChild(span);
    }
    
    if (cursorOwned || _streamingTarget === el) {
        _streamingTarget = el;
        el.appendChild(_getOrCreateCursor());
    }
}

// ── thought-panel helpers ──────────────────────────────────────────────────────

/**
 * Append a streamed reasoning token to the thought panel and the full-size Reasoning tab.
 *
 * The token count is only an estimate (about 4 characters per token). The pulse dot shows that thinking is in progress.
 * @param {string} token
 */
function appendThoughtToken(token) {
    // Step 1: Initialize function scope and validate execution context
    const str = String(token || '');
    if (!str) return;
    _thoughtTokenCount += Math.ceil(str.length / 4);
    const thoughtsEl = $('introspectionThoughts');
    if (thoughtsEl) {
        thoughtsEl.textContent += str;
        _triggerScroll(thoughtsEl);
    }
    const countEl = $('thoughtsTokenCount');
    if (countEl) countEl.textContent = `~${_thoughtTokenCount.toLocaleString()} tokens`;
    const dot = $('thoughtsPulseDot');
    if (dot && !dot.classList.contains('active')) dot.classList.add('active');
    // Mirror content to the full-view reasoning panel (Reasoning tab)
    const fullView = $('reasoningFullView');
    if (fullView) {
        fullView.textContent += str;
        _triggerScroll(fullView);
    }
}

/**
 * Clear all reasoning panels and counters before a new response starts.
 */
function resetThoughtsPanel() {
    // Step 1: Initialize function scope and validate execution context
    _thoughtTokenCount = 0;
    const thoughtsEl = $('introspectionThoughts');
    if (thoughtsEl) thoughtsEl.textContent = '';
    const countEl = $('thoughtsTokenCount');
    if (countEl) countEl.textContent = '0 tokens';
    const dot = $('thoughtsPulseDot');
    if (dot) dot.classList.remove('active');
    const box = $('introspectionBox');
    if (box) box.classList.remove('active');
    const fullView = $('reasoningFullView');
    if (fullView) fullView.textContent = '';
}

/**
 * Stop the "thinking" pulse animation once the reasoning phase is over.
 */
function finalizeThoughts() {
    // Step 1: Initialize function scope and validate execution context
    const dot = $('thoughtsPulseDot');
    if (dot) dot.classList.remove('active');
}

// ── model / download lists ─────────────────────────────────────────────────────

/**
 * Build the HTML for one model card. The title is escaped; ``metaHtml`` and ``buttonHtml`` are inserted as-is,
 * so callers must escape any untrusted values they put in them.
 * @param {string} title
 * @param {string} metaHtml
 * @param {string} buttonHtml
 * @returns {string}
 */
function buildModelCard(title, metaHtml, buttonHtml) {
    // Step 1: Initialize function scope and validate execution context
    return `<div class="model-card">
        <div class="model-title">${escapeHtml(title)}</div>
        <div class="model-meta">${metaHtml}</div>
        ${buttonHtml}
    </div>`;
}

/**
 * Draw the model list with an Active / Set Active / "Download in Karl" button per model.
 *
 * Downloading happens in the desktop app, so a missing model only gets a disabled hint button.
 * @param {any[]} models
 */
function renderModels(models) {
    // Step 1: Initialize function scope and validate execution context
    if (!Array.isArray(models)) models = [];
    $('modelList').innerHTML = models.length ? models.map(model => {
        if (!model) return '';
        const action = model.active
            ? '<button disabled>Active</button>'
            : model.installed
                ? `<button data-model="${escapeHtml(model.filename || '')}">Set Active</button>`
                : '<button disabled>Download in Karl</button>';
        const meta = `${escapeHtml(model.filename || '')}<br>${model.n_ctx || '--'} ctx · ${model.min_ram_gb || '--'} GB RAM · ${model.active ? 'Active' : model.installed ? 'Installed' : 'Missing'}`;
        return buildModelCard(model.name || model.filename || 'Unknown model', meta, action);
    }).join('') : '<div class="model-card"><div class="model-meta">No model registry entries found.</div></div>';
}

/**
 * Show the static model-size tiers (1.5B, 7B/8B, 14B, 70B) and whether a matching model is already installed.
 *
 * A tier counts as installed if any installed model name contains the tier's first word, case-insensitively
 * (a rough match, not an exact one).
 * @param {any[]} models
 */
function renderDownloadRegistry(models) {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('downloadRegistry');
    if (!panel) return;
    const installed = new Set((models || []).filter(model => model.installed).map(model => model.name || model.filename));
    const tiers = [
        ['1.5B Qwen', 'Fast local scout tier for low-RAM systems and short context tests.', '4 GB'],
        ['7B/8B Qwen/LLaMA', 'Balanced daily-driver tier for coding, review, and RAG workflows.', '12-16 GB'],
        ['14B', 'Higher reasoning headroom for deep planning and longer workbench sessions.', '24 GB'],
        ['70B', 'Maximum local quality tier for workstation-class systems.', '64+ GB']
    ];
    panel.innerHTML = tiers.map(([name, description, ram]) => {
        const isInstalled = Array.from(installed).some(item => String(item).toLowerCase().includes(name.split(' ')[0].toLowerCase()));
        const action = `<button disabled>${isInstalled ? 'Available' : 'Bridge Download Required'}</button>`;
        const meta = `${escapeHtml(description)}<br>Recommended RAM: ${escapeHtml(ram)} · ${isInstalled ? 'Installed candidate detected' : 'Download requires desktop bridge support'}`;
        return buildModelCard(name, meta, action);
    }).join('');
}

/**
 * Fill the LoRA adapter dropdown with a "None" entry plus each adapter; the active adapter is pre-selected.
 * @param {string[]} adapters
 * @param {string} activeAdapter
 */
function renderAdapters(adapters, activeAdapter) {
    // Step 1: Initialize function scope and validate execution context
    if (!Array.isArray(adapters)) adapters = [];
    const select = $('adapterSelect');
    if (!select) return;
    
    select.innerHTML = '<option value="">None (Baseline Model)</option>' + 
        adapters.map(adapter => {
            const selected = adapter === activeAdapter ? 'selected' : '';
            return `<option value="${escapeHtml(adapter)}" ${selected}>${escapeHtml(adapter)}</option>`;
        }).join('');
}

// ── theme catalog ──────────────────────────────────────────────────────────────

/**
 * Draw the clickable theme preview cards (four colour swatches each) and highlight the selected theme.
 */
function renderThemeCatalog() {
    // Step 1: Initialize function scope and validate execution context
    const grid = $('themeGrid');
    if (!grid) return;
    const themes = Array.isArray(window.KARL_THEMES) ? window.KARL_THEMES : [];
    grid.innerHTML = themes.map(theme => {
        if (!theme) return '';
        const vars = theme.vars || {};
        const active = $('themeSelect').value === theme.id;
        const bg = vars['--karl-bg'] || '';
        const panel = vars['--karl-panel'] || '';
        const border = vars['--karl-border'] || '';
        const accent = vars['--karl-accent'] || '';
        return `
            <div class="theme-card ${active ? 'active' : ''}" data-theme-id="${escapeHtml(theme.id)}">
                <div class="theme-card-title">${escapeHtml(theme.name)}</div>
                <div class="theme-card-colors">
                    <span style="background:${escapeHtml(bg)}" title="BG"></span>
                    <span style="background:${escapeHtml(panel)}" title="Panel"></span>
                    <span style="background:${escapeHtml(border)}" title="Border"></span>
                    <span style="background:${escapeHtml(accent)}" title="Accent"></span>
                </div>
                <div class="theme-card-desc">${escapeHtml(theme.description)}</div>
            </div>
        `;
    }).join('');
}

// ── pending-edit cards ─────────────────────────────────────────────────────────

/**
 * Format an edit's line-count change, e.g. "+12 lines" or "-3 lines".
 * @param {any} edit
 */
function formatLineDelta(edit) {
    // Step 1: Initialize function scope and validate execution context
    if (!Number.isFinite(edit.lineDelta)) return 'line delta unknown';
    return `${edit.lineDelta >= 0 ? '+' : ''}${edit.lineDelta} lines`;
}

/**
 * Classify an edit as high / medium / low risk by size.
 *
 * High: more than 400 changed lines or 120,000 bytes; medium: over 80 lines or 30,000 bytes.
 * @param {any} edit
 * @returns {string} HTML span
 */
function riskLabel(edit) {
    // Step 1: Initialize function scope and validate execution context
    const delta = Math.abs(Number(edit.lineDelta || 0));
    if (delta > 400 || Number(edit.bytes || 0) > 120000) return '<span class="risk high">high risk</span>';
    if (delta > 80 || Number(edit.bytes || 0) > 30000) return '<span class="risk medium">medium risk</span>';
    return '<span class="risk low">low risk</span>';
}

/**
 * Draw the review queue of proposed file edits with Preview/Apply/Reject/Open/Copy/Rollback buttons.
 *
 * The status filter limits what is shown. Buttons carry the edit id in data-* attributes and are
 * handled by click delegation elsewhere.
 */
function renderPendingEdits() {
    // Step 1: Initialize function scope and validate execution context
    const queue = $('changeQueue');
    const filter = $('changeFilter') ? $('changeFilter').value : '';
    const edits = Array.from(pendingEdits.values()).filter(edit => !filter || edit.status === filter);
    if (!edits.length) {
        queue.className = 'queue empty';
        queue.innerText = pendingEdits.size ? 'No edits match the current filter.' : 'No pending Karl edits.';
        return;
    }
    queue.className = 'queue';
    queue.innerHTML = edits.map(edit => `
        <div class="change-card" style="margin-bottom: 8px;">
            <div class="change-title" style="font-size: 13px;">${escapeHtml(edit.filename || 'unknown')}</div>
            <div class="change-meta" style="margin-top: 4px; line-height: 1.4;">
                Path: <strong style="font-family:monospace; font-size:10px;">${escapeHtml(edit.filepath || '')}</strong><br>
                Size: <strong>${Number(edit.bytes || 0)} bytes</strong> · Delta: <strong>${formatLineDelta(edit)}</strong> · Risk: ${riskLabel(edit)} · Status: <span class="state-chip ${escapeHtml(edit.status || 'proposed')}">${escapeHtml(edit.status || 'proposed')}</span><br>
                Summary: <em>${escapeHtml(edit.summary || 'Proposed Karl edit')}</em>
            </div>
            <div class="change-actions" style="margin-top:8px;">
                <button data-preview="${escapeHtml(edit.id)}">Preview Diff</button>
                <button class="primary" data-apply="${escapeHtml(edit.id)}">Apply File</button>
                <button class="danger" data-reject="${escapeHtml(edit.id)}">Reject File</button>
                <button data-open="${escapeHtml(edit.id)}">Open File</button>
                <button data-copy-path="${escapeHtml(edit.id)}">Copy Path</button>
                <button data-rollback="${escapeHtml(edit.id)}">Rollback</button>
            </div>
        </div>
    `).join('');
}

// ── lab diff ───────────────────────────────────────────────────────────────────

/**
 * Show a character-level diff of the two prompt-lab outputs (deleted text in red, added in green).
 *
 * Greedy algorithm: identical characters pass through; on a mismatch, text is skipped in A then in B
 * until the two sides line up again. It is simple but not a minimal diff.
 * @param {string} textA
 * @param {string} textB
 */
function renderLabDiff(textA, textB) {
    // Step 1: Initialize function scope and validate execution context
    const diffContainer = $('labDiff');
    if (!textA || !textB) {
        diffContainer.innerText = 'Run both prompts before computing a diff.';
        return;
    }
    let i = 0, j = 0;
    let html = '';
    while (i < textA.length || j < textB.length) {
        if (i < textA.length && j < textB.length && textA[i] === textB[j]) {
            html += escapeHtml(textA[i]);
            i++; j++;
        } else {
            let del = '', add = '';
            while (i < textA.length && (j >= textB.length || textA[i] !== textB[j])) { del += textA[i]; i++; }
            while (j < textB.length && (i >= textA.length || textA[i] !== textB[j])) { add += textB[j]; j++; }
            if (del) html += `<span class="diff-del">${escapeHtml(del)}</span>`;
            if (add) html += `<span class="diff-add">${escapeHtml(add)}</span>`;
        }
    }
    diffContainer.innerHTML = html;
}

// ── prompt pairs ───────────────────────────────────────────────────────────────

/**
 * Fill the saved prompt-pair dropdown.
 * @param {{name: string}[]} pairs
 */
function renderPromptPairs(pairs) {
    // Step 1: Initialize function scope and validate execution context
    $('promptPairSelect').innerHTML = '<option value="">Saved prompt pairs...</option>' + pairs.map(pair => {
        return `<option value="${escapeHtml(pair.name)}">${escapeHtml(pair.name)}</option>`;
    }).join('');
}

// ── knowledge base ─────────────────────────────────────────────────────────────

/**
 * Update the knowledge-base counters, source list and source filter from a backend snapshot.
 * @param {any} snapshot
 */
function renderKbSnapshot(snapshot) {
    // Step 1: Initialize function scope and validate execution context
    if (!snapshot || typeof snapshot !== 'object') snapshot = {};
    lastKbSnapshot = snapshot;
    const sources = Array.isArray(snapshot.sources) ? snapshot.sources : [];
    $('kbSourceCount').innerText = snapshot.total_sources ?? sources.length ?? 0;
    $('kbChunkCount').innerText = snapshot.total_chunks ?? 0;
    $('kbIngestState').innerText = snapshot.ingesting ? 'Running' : 'Ready';
    $('kbSourceList').innerHTML = sources.length ? sources.map(source => `
        <div class="source-item ${source && source.name === kbSelectedSource ? 'active' : ''}" data-source="${escapeHtml(source ? source.name : '')}">
            <span class="source-name">${escapeHtml(source ? source.name : 'unknown')}</span>
            <span>${Number(source ? source.chunks : 0)} chunks</span>
        </div>
    `).join('') : '<div class="source-item">No indexed sources yet.</div>';

    $('kbSourceFilter').innerHTML = '<option value="">All sources</option>' + sources.map(source => {
        return `<option value="${escapeHtml(source ? source.name : '')}">${escapeHtml(source ? source.name : 'unknown')}</option>`;
    }).join('');
    if (kbSelectedSource) $('kbSourceFilter').value = kbSelectedSource;
}

/**
 * Draw the list of files queued for batch knowledge-base ingestion.
 */
function renderKbQueue() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('kbQueue');
    if (!kbQueue.length) {
        panel.className = 'queue-list empty';
        panel.innerText = 'No files queued for batch ingest.';
        return;
    }
    panel.className = 'queue-list';
    panel.innerHTML = kbQueue.map((item, index) => `
        <div class="queue-row ${escapeHtml(item.status || 'queued')}">
            <span>${escapeHtml(item.path)}</span>
            <strong>${escapeHtml(item.status || 'queued')}</strong>
            <button data-remove-kb="${index}">Remove</button>
        </div>
    `).join('');
}

/**
 * Show knowledge-base search hits (rank, source, chunk, distance and text capped at 1800 characters).
 *
 * The results are saved in ``lastKbResults`` so "Send to Chat/Swarm" buttons can look them up by index.
 * @param {any} payload
 */
function renderKbSearch(payload) {
    // Step 1: Initialize function scope and validate execution context
    if (!payload || typeof payload !== 'object') payload = {};
    renderKbSnapshot(payload.snapshot || {});
    const results = Array.isArray(payload.results) ? payload.results : [];
    lastKbResults = results;
    $('kbResults').innerHTML = results.length ? results.map((result, index) => {
        if (!result) result = {};
        return `
        <div class="result-card">
            <div class="result-meta">Rank ${escapeHtml(result.rank ?? index)} · ${escapeHtml(result.source_file || 'unknown')} · Chunk ${escapeHtml(result.chunk_id || '0')} · dist=${Number(result.distance || 0).toFixed(4)}</div>
            <pre>${escapeHtml(result.text || '').slice(0, 1800)}</pre>
            <div class="action-row compact-actions" style="margin-top:6px;">
                <button data-send-chat="${index}">Send to Chat</button>
                <button data-send-swarm="${index}">Send to Swarm</button>
            </div>
        </div>
    `; }).join('') : '<div class="result-card">No chunks matched the current query and threshold.</div>';
}

// ── codex ──────────────────────────────────────────────────────────────────────

/**
 * List the codex (reference chapter) topics.
 * @param {string[]} topics
 */
function renderCodexTopics(topics) {
    // Step 1: Initialize function scope and validate execution context
    if (!Array.isArray(topics)) topics = [];
    $('codexList').innerHTML = topics.length ? topics.map(topic => {
        return `<div class="source-item" data-topic="${escapeHtml(topic)}"><span>${escapeHtml(topic)}</span></div>`;
    }).join('') : '<div class="source-item">No chapters loaded.</div>';
}

// ── branches ───────────────────────────────────────────────────────────────────

/**
 * Draw the conversation-branch tree; the active branch is highlighted.
 */
function renderBranches() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('branchTree');
    if (!panel) return;
    if (!conversationBranches.length) {
        panel.className = 'branch-tree empty';
        panel.innerText = 'No conversation branches yet.';
        return;
    }
    panel.className = 'branch-tree';
    panel.innerHTML = conversationBranches.map(branch => `
        <button class="branch-node ${branch.id === activeBranchId ? 'active' : ''}" data-branch="${escapeHtml(branch.id)}">
            <span>${escapeHtml(branch.title || 'Untitled branch')}</span>
            <strong>${branch.turns.length} turn${branch.turns.length === 1 ? '' : 's'}</strong>
        </button>
    `).join('');
}

// ── task queue ─────────────────────────────────────────────────────────────────

/**
 * Draw up to 12 queued tasks with their status.
 */
function renderTaskQueue() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('taskQueue');
    if (!taskQueue.length) {
        panel.className = 'task-queue empty';
        panel.innerText = 'No tasks queued.';
        return;
    }
    panel.className = 'task-queue';
    panel.innerHTML = taskQueue.slice(0, 12).map(task => `
        <div class="task-item ${escapeHtml(task.status)}">
            <strong>${escapeHtml(task.mode)}</strong>
            <span>${escapeHtml(task.status)} · ${escapeHtml(task.objective)}</span>
        </div>
    `).join('');
}

// ── context meter ──────────────────────────────────────────────────────────────

/**
 * Show how much of the context was sent (e.g. "1200/80000 chars") and warn if it was truncated.
 * @param {any} meta {sentChars, originalChars, label, truncated} or null
 */
function renderContextMeta(meta) {
    // Step 1: Initialize function scope and validate execution context
    if (!meta) {
        $('contextMeter').innerText = 'Context package: none queued.';
        $('contextMeter').className = 'context-meter';
        return;
    }
    $('contextMeter').innerText = `Context package: ${meta.sentChars}/${meta.originalChars} chars sent from ${meta.label}${meta.truncated ? ' · safely truncated' : ''}.`;
    $('contextMeter').className = `context-meter ${meta.truncated ? 'warn' : 'ok'}`;
}

// ── swarm timeline / log ───────────────────────────────────────────────────────

/**
 * Add an entry to the swarm timeline (newest first) and to the Swarm Progress feed (with a time stamp).
 * @param {string} title
 * @param {string} detail
 */
function addTimeline(title, detail) {
    // Step 1: Initialize function scope and validate execution context
    const item = document.createElement('div');
    item.className = 'timeline-item';
    item.innerHTML = `<strong>${escapeHtml(title)}</strong><br>${escapeHtml(detail)}`;
    $('timeline').prepend(item);
    // Mirror to the Swarm Progress tab feed
    const feed = $('swarmFeed');
    if (feed) {
        const placeholder = feed.querySelector('[data-swarm-empty]');
        if (placeholder) placeholder.remove();
        const now = new Date();
        const t = now.toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const entry = document.createElement('div');
        entry.className = 'swarm-entry';
        entry.innerHTML = `
            <span class="swarm-entry-time">${escapeHtml(t)}</span>
            <div class="swarm-entry-text">
                <strong>${escapeHtml(title)}</strong>
                <span>${escapeHtml(detail)}</span>
            </div>`;
        feed.appendChild(entry);
        _triggerScroll(feed);
    }
}

/**
 * Append a line to the terminal-style log panel and keep it scrolled to the bottom.
 * @param {string} message
 */
function log(message) {
    // Step 1: Initialize function scope and validate execution context
    const terminal = $('terminal');
    terminal.innerText += `\n${message}`;
    _triggerScroll(terminal);
}

// ── diagnostics ────────────────────────────────────────────────────────────────

/**
 * List editor problems grouped by file; severity decides the colour class.
 *
 * Each row stores file/line/column in data attributes so a click can open that location.
 * @param {any} diagDetails
 */
function renderDiagnosticsList(diagDetails) {
    // Step 1: Initialize function scope and validate execution context
    const listPanel = $('diagnosticsList');
    if (!listPanel) return;
    if (!diagDetails || !diagDetails.files || !Object.keys(diagDetails.files).length) {
        listPanel.innerHTML = '<div class="source-item">No current diagnostics found.</div>';
        return;
    }
    let html = '';
    Object.entries(diagDetails.files).forEach(([filepath, items]) => {
        const basename = filepath.split(/[/\\]/).pop();
        html += `<div class="diagnostics-file-header" style="font-weight:750; margin: 8px 0 4px; color: var(--karl-accent-2);">${escapeHtml(basename)}</div>`;
        items.forEach(item => {
            const severityClass = item.severity === 'error' ? 'risk high' : item.severity === 'warning' ? 'risk medium' : 'risk low';
            html += `
                <div class="source-item" data-diag-file="${escapeHtml(filepath)}" data-diag-line="${item.line}" data-diag-char="${item.character}" style="grid-template-columns: auto minmax(0, 1fr) auto; gap: 6px; padding: 4px 6px;">
                    <span class="${severityClass}" style="padding: 1px 4px; font-size: 8px;">${escapeHtml(item.severity)}</span>
                    <span style="font-size: 10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(item.message)}">${escapeHtml(item.message)}</span>
                    <strong style="font-size: 9px; color: var(--karl-muted);">L${item.line}</strong>
                </div>
            `;
        });
    });
    listPanel.innerHTML = html;
}

// ── cockpit state ──────────────────────────────────────────────────────────────

/**
 * Refresh the cockpit summary (workspace, active file, git branch, pending edits and diagnostics counts).
 * @param {any} state
 */
function updateCockpitState(state) {
    // Step 1: Initialize function scope and validate execution context
    $('cockpitWorkspace').innerText = state.workspacePath || '--';
    $('cockpitFile').innerText = state.activeFile ? state.activeFile.split(/[/\\]/).pop() : 'none';
    $('cockpitFile').title = state.activeFile || '';
    $('cockpitBranch').innerText = state.gitBranch || '--';
    $('gitBranchDisplay').innerText = state.gitBranch || '--';
    $('cockpitPendingChanges').innerText = state.pendingEditsCount || 0;

    const diag = state.diagnostics || { error: 0, warning: 0, info: 0, hint: 0 };
    const totalDiag = diag.error + diag.warning + diag.info + diag.hint;
    $('cockpitDiagnostics').innerText = `${totalDiag} problem${totalDiag === 1 ? '' : 's'} (${diag.error} errs)`;

    $('diagErrorsCount').innerText = diag.error || 0;
    $('diagWarningsCount').innerText = diag.warning || 0;
    $('diagInfosCount').innerText = diag.info || 0;
    $('diagHintsCount').innerText = diag.hint || 0;

    renderDiagnosticsList(state.diagnosticsDetails);
}

// ── runtime status ─────────────────────────────────────────────────────────────

/**
 * Reset all runtime fields to their offline placeholders.
 */
function renderRuntimeOffline() {
    // Step 1: Initialize function scope and validate execution context
    $('runtimeModel').innerText = 'unknown';
    $('runtimeState').innerText = 'offline';
    $('runtimeAdapter').innerText = 'none';
    $('runtimeSystem').innerText = '--';
    $('cockpitModel').innerText = 'unknown';
    $('cockpitSystem').innerText = '--';
}

/**
 * Show the backend's model, adapter, memory, client count and bridge capabilities, and update the connection label.
 *
 * The connection label shows "Busy" while generating, or the latency in ms; more than 200 ms counts as slow.
 * The RAM check compares system RAM with the model's minimum.
 * @param {any} status
 * @param {number} [latency] round-trip time in ms
 */
function renderRuntimeStatus(status, latency = 0) {
    // Step 1: Initialize function scope and validate execution context
    if (!status || typeof status !== 'object') status = {};
    const model = status.model || {};
    const adapter = status.adapter || {};
    const runtime = status.runtime || {};
    const system = status.system || {};
    const bridge = status.bridge || {};
    // Defensive parsing: every field may be missing from an older or partial backend response.
    const clients = Number.isInteger(bridge.clients) ? bridge.clients : 0;

    const desc = `${model.name || 'none'}${model.loaded ? ' loaded' : ''}`;
    $('runtimeModel').innerText = desc;
    $('cockpitModel').innerText = desc;
    activeModelFilename = model.name || '';

    const stateStr = `${runtime.state || 'idle'} · ${clients} client${clients === 1 ? '' : 's'}`;
    $('runtimeState').innerText = stateStr;
    const adapterName = adapter.name || '';
    $('runtimeAdapter').innerText = adapterName || 'none';
    if ($('adapterSelect')) {
        $('adapterSelect').value = adapterName;
    }

    const sysStr = `${system.ram_mb ?? '--'} MB · ${model.n_ctx || '--'} ctx`;
    $('runtimeSystem').innerText = sysStr;
    $('cockpitSystem').innerText = sysStr;

    if ($('systemActiveModel')) $('systemActiveModel').innerText = model.name || 'none';
    if ($('systemContext')) $('systemContext').innerText = `${model.n_ctx || '--'} ctx`;
    if ($('systemRamCheck')) {
        const ramGb = Number(system.ram_mb || 0) / 1024;
        const needGb = Number(model.min_ram_gb || 0);
        $('systemRamCheck').innerText = needGb && ramGb ? (ramGb >= needGb ? 'ok' : 'low ram') : 'unknown';
    }
    if ($('systemAdapterWarning')) {
        $('systemAdapterWarning').innerText = adapter.name && adapter.base_model && model.name && adapter.base_model !== model.name
            ? 'base mismatch' : 'none';
    }
    if (status.bridge && status.bridge.version) {
        $('bridgeMeta').dataset.version = status.bridge.version;
    }
    // Without the "vision" capability in the bridge, the image tools show a warning.
    if (status.bridge && status.bridge.capabilities && status.bridge.capabilities.includes('vision')) {
        $('visionBridgeWarning').style.display = 'none';
    } else {
        $('visionBridgeWarning').style.display = 'block';
    }

    // Only rewrite the label while connected, so this late status reply cannot overwrite an offline or error state.
    // Dynamic latency and workload status indication
    if (connectionState === 'connected' || connectionState === 'busy') {
        const isBusy = runtime.state === 'running' || runtime.state === 'generating';
        const isHighLatency = latency > 200;
        
        let label = 'Connected';
        let stateClass = 'connected';

        if (isBusy) {
            label = 'Connected [Busy]';
            stateClass = 'busy';
        } else if (isHighLatency) {
            label = `Connected (${latency}ms)`;
            stateClass = 'busy';
        } else if (latency > 0) {
            label = `Connected (${latency}ms)`;
            stateClass = 'connected';
        }
        
        connectionState = stateClass;
        $('statusDot').className = `status-dot ${stateClass}`;
        $('statusText').innerText = label;
        $('cockpitConnection').innerText = label.toLowerCase();
        $('cockpitConnection').className = `state-chip ${stateClass}`;
    }
}

// ── chat bubbles ───────────────────────────────────────────────────────────────

/**
 * Add a chat bubble for ``role`` ("user" or "assistant").
 *
 * An empty assistant bubble starts a streaming response and gets the blinking cursor.
 * @param {string} role
 * @param {string} text
 */
function appendMessageBubble(role, text) {
    // Step 1: Initialize function scope and validate execution context
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `<div class="message-role">${role === 'user' ? 'User' : 'Karl'}</div><div class="message-content"></div>`;
    const contentEl = msg.querySelector('.message-content');
    if (text) {
        contentEl.innerText = text;
    } else if (role === 'assistant') {
        // Blank assistant bubble signals start of a streaming response.
        // Attach the animated cursor so the user sees immediate activity.
        attachStreamingCursor(contentEl);
    }
    $('chatMessages').appendChild(msg);
    _triggerScroll($('chatMessages'));
}

/**
 * Add a streamed token to the newest assistant bubble, routing any <think> text to the reasoning panel.
 * @param {string} token
 */
function appendChatToken(token) {
    // Step 1: Initialize function scope and validate execution context
    const target = $('chatMessages').querySelector('.message.assistant:last-child .message-content');
    if (target) {
        routeThinkMarkup(token, target);
        _triggerScroll($('chatMessages'));
    }
}

// Routes incoming response-stream tokens: text content goes to the chat
// bubble; <think>…</think> sections are routed to the reasoning panel.
// The streaming cursor is kept at the very end of chat content at all times.
// Detail: a small state machine driven by ``responseThinkActive``. NOTE: it only recognises <think> and
// </think> tags that arrive whole inside one token.
function routeThinkMarkup(token, chatTarget) {
    // Step 1: Initialize function scope and validate execution context
    const text = String(token || '');
    // Fast path: ordinary text with no tags goes straight to the bubble.
    if (!responseThinkActive && !text.includes('<think') && !text.includes('</think>')) {
        _appendChatText(chatTarget, text);
        return;
    }
    // Slow path: split the token at tag boundaries and send each part to the right place.
    let remaining = text;
    while (remaining.length) {
        if (responseThinkActive) {
            const close = remaining.indexOf('</think>');
            $('introspectionBox').classList.add('active');
            if (close === -1) {
                appendThoughtToken(remaining);
                remaining = '';
            } else {
                appendThoughtToken(remaining.slice(0, close));
                remaining = remaining.slice(close + '</think>'.length);
                responseThinkActive = false;
            }
            continue;
        }
        const open = remaining.indexOf('<think>');
        if (open === -1) {
            _appendChatText(chatTarget, remaining);
            break;
        }
        _appendChatText(chatTarget, remaining.slice(0, open));
        remaining = remaining.slice(open + '<think>'.length);
        responseThinkActive = true;
    }
}

// ── quick actions / recent tasks ───────────────────────────────────────────────

/**
 * Draw the one-click workflow buttons (names match the ids in commands.js WORKFLOW_REGISTRY).
 */
function renderQuickActions() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('quickActions');
    if (!panel) return;
    const actions = [
        { label: '⚡ Fix Selection', workflow: 'fixSelection' },
        { label: '🔍 Explain Selection', workflow: 'explainSelection' },
        { label: '🧪 Generate Tests', workflow: 'generateTests' },
        { label: '📋 Review File', workflow: 'reviewActiveFile' },
        { label: '📤 Send to Swarm', workflow: 'sendCurrentFileToSwarm' },
        { label: '💬 Ask Workspace', workflow: 'askWorkspace' },
    ];
    panel.innerHTML = actions.map(action => `
        <button class="quick-action" data-workflow="${escapeHtml(action.workflow)}">${escapeHtml(action.label)}</button>
    `).join('');
}

/**
 * List the 8 most recent tasks; clicking one re-runs it (via data-task-idx).
 */
function renderRecentTasks() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('recentTasksHistory');
    if (!panel) return;
    if (!recentTasks.length) {
        panel.className = 'recent-list empty';
        panel.innerText = 'No recent tasks.';
        return;
    }
    panel.className = 'recent-list';
    panel.innerHTML = recentTasks.slice(0, 8).map((task, index) => `
        <button class="recent-item" data-task-idx="${index}">
            <strong>${escapeHtml(task.title || task.workflowId)}</strong>
            <span>${escapeHtml((task.objective || '').slice(0, 60))}</span>
        </button>
    `).join('');
}

/**
 * List the 6 most recent knowledge-base searches; clicking one repeats it.
 */
function renderRecentKbQueries() {
    // Step 1: Initialize function scope and validate execution context
    const panel = $('recentKbQueries');
    if (!panel) return;
    if (!recentKbQueries.length) {
        panel.className = 'recent-list empty';
        panel.innerText = 'No recent queries.';
        return;
    }
    panel.className = 'recent-list';
    panel.innerHTML = recentKbQueries.slice(0, 6).map((q, index) => `
        <button class="recent-item" data-kb-query="${index}">
            <span>${escapeHtml(q)}</span>
        </button>
    `).join('');
}
