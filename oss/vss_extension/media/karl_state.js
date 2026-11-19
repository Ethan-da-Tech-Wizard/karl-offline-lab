// ==============================================================================
// Program: karl_state.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: State container tracking active conversation sessions, model settings, and workspace diffs.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// acquireVsCodeApi() may be called only once per webview; this handle is how the page
// sends messages to the extension host (see sidebarProvider.js).
const vscode = acquireVsCodeApi();
// Initial data (saved state, workspace folder, port...) injected by the extension host before this script runs.
const boot = window.KARL_BOOTSTRAP || {};

// --- Shared mutable state ---
// These are script-level globals used across karl_state.js, karl_socket.js, karl_render.js
// and karl.js (all loaded as plain scripts, not modules). Connection first:
let socket = null;
let reconnectTimer = null;
let runtimeStatusTimer = null;
let bridgeMetaTimer = null;
// Chat, lab (prompt A/B), knowledge-base and swarm state:
let chatFinished = true;
let labRunning = false;
let currentLabTarget = '';
let labOutputA = '';
let labOutputB = '';
let kbSelectedSource = '';
let connectionState = 'offline';
let pendingEdits = new Map();
let swarmRunHistory = [];
let lastSwarmRagResults = [];
let manualDisconnect = false;
let persistTimer = null;
let recentTasks = [];
let recentKbQueries = [];
let taskQueue = [];
let activeTaskId = '';
let activeWorkflowId = '';
let lastHeartbeatAt = null;
let lastConnectedAt = null;
let lastBridgeError = '';
let kbQueue = [];
let kbQueueRunning = false;
// Conversation branches and the response currently being streamed:
let conversationBranches = [];
let activeBranchId = '';
let currentConversationInput = '';
let responseThinkActive = false;
let currentResponseText = '';
// Reconnect backoff and last-seen knowledge-base data:
let nextReconnectSec = 5;
let lastKbSnapshot = {};
let lastKbResults = [];
let swarmAgentProfiles = {};
let currentTargetFilepath = '';
let activeModelFilename = '';

/**
 * Shorthand for document.getElementById.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
const $ = (id) => document.getElementById(id);

/**
 * Fill the UI from the state the extension saved last time (``boot.persisted``).
 *
 * Every list is validated (type checks, length caps) because the saved data may come from an
 * older version or be hand-edited. Then applies the look, opens the last tab and starts a
 * 1-second timer that refreshes the bridge status display.
 */
function hydrate() {
    // Step 1: Initialize function scope and validate execution context
    const persisted = boot.persisted || {};
    $('workspace').value = persisted.workspace || boot.workspaceFolder || '';
    $('bridgePort').value = persisted.port || boot.port || 8080;
    $('taskMode').value = persisted.taskMode || 'Custom Task';
    $('themeSelect').value = persisted.theme || 'obsidian-core';
    $('customAccent').value = persisted.customAccent || themeById($('themeSelect').value).vars['--karl-accent'];
    $('layoutSelect').value = persisted.layout || 'cockpit';
    
    // Aesthetic settings checkboxes
    $('syncVsCodeTheme').checked = persisted.syncVsCodeTheme !== false;
    $('highContrastMode').checked = !!persisted.highContrastMode;
    $('reducedMotion').checked = !!persisted.reducedMotion;
    $('animationIntensity').value = persisted.animationIntensity !== undefined ? persisted.animationIntensity : 100;

    // Defensive filtering: keep only entries with the expected field types, then cap the count.
    const rawTasks = Array.isArray(persisted.recentTasks) ? persisted.recentTasks : [];
    recentTasks = rawTasks.filter(t => t && typeof t.workflowId === 'string' && typeof t.title === 'string' && typeof t.objective === 'string').slice(0, 15);

    const rawQueries = Array.isArray(persisted.recentKbQueries) ? persisted.recentKbQueries : [];
    recentKbQueries = rawQueries.filter(q => typeof q === 'string').slice(0, 12);

    const rawBranches = Array.isArray(persisted.conversationBranches) ? persisted.conversationBranches : [];
    conversationBranches = rawBranches.filter(b => b && typeof b.id === 'string' && typeof b.title === 'string' && Array.isArray(b.turns)).slice(0, 20);
    
    activeBranchId = persisted.activeBranchId || (conversationBranches[0] && conversationBranches[0].id) || '';
    
    applyAppearance();
    if (persisted.workspaceTab) {
        switchWorkspace(persisted.workspaceTab);
    } else {
        switchWorkspace('cockpit');
    }
    // Guard so hydrating twice does not start a second timer.
    if (!bridgeMetaTimer) {
        bridgeMetaTimer = setInterval(updateBridgeMeta, 1000);
    }
}

/**
 * Save UI state to the extension host, debounced by 250 ms.
 *
 * Clearing the previous timer means a burst of changes (typing, sliders) sends only one message.
 */
function persist() {
    // Step 1: Initialize function scope and validate execution context
    clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
        vscode.postMessage({
            command: 'persist_state',
            state: {
                workspace: $('workspace').value,
                port: Number($('bridgePort').value) || boot.port || 8080,
                taskMode: $('taskMode').value,
                // Remember which tab is open; falls back to "cockpit" when none is marked active.
                workspaceTab: document.querySelector('.tab.active')?.dataset.workspace || 'cockpit',
                theme: $('themeSelect').value,
                customAccent: $('customAccent').value,
                layout: $('layoutSelect').value,
                syncVsCodeTheme: $('syncVsCodeTheme').checked,
                highContrastMode: $('highContrastMode').checked,
                reducedMotion: $('reducedMotion').checked,
                animationIntensity: Number($('animationIntensity').value),
                // Same caps as in hydrate(), so stored state cannot grow without limit.
                recentTasks: recentTasks.slice(0, 15),
                recentKbQueries: recentKbQueries.slice(0, 12),
                conversationBranches: conversationBranches.slice(0, 20),
                activeBranchId
            }
        });
    }, 250);
}

/**
 * Populate the theme and layout dropdowns from window.KARL_THEMES / KARL_LAYOUTS and draw the theme catalog.
 */
function initializeAppearance() {
    // Step 1: Initialize function scope and validate execution context
    $('themeSelect').innerHTML = window.KARL_THEMES.map(theme => {
        return `<option value="${escapeHtml(theme.id)}">${escapeHtml(theme.name)}</option>`;
    }).join('');
    $('layoutSelect').innerHTML = window.KARL_LAYOUTS.map(layout => {
        return `<option value="${escapeHtml(layout.id)}">${escapeHtml(layout.name)}</option>`;
    }).join('');
    renderThemeCatalog();
}

/**
 * Look up a theme by id; falls back to the first theme, then to a built-in default so callers never get undefined.
 * @param {string} id
 */
function themeById(id) {
    // Step 1: Initialize function scope and validate execution context
    const themes = Array.isArray(window.KARL_THEMES) ? window.KARL_THEMES : [];
    return themes.find(theme => theme && theme.id === id) || themes[0] || { id: 'obsidian-core', vars: { '--karl-accent': '#00c2ff' }, name: 'Obsidian Core', description: '' };
}

/**
 * Look up a layout by id with the same fallback chain as themeById.
 * @param {string} id
 */
function layoutById(id) {
    // Step 1: Initialize function scope and validate execution context
    const layouts = Array.isArray(window.KARL_LAYOUTS) ? window.KARL_LAYOUTS : [];
    return layouts.find(layout => layout && layout.id === id) || layouts[0] || { id: 'cockpit', name: 'Cockpit', description: '' };
}

/**
 * Apply the selected theme, accent colour, layout and accessibility options to the page, then save.
 *
 * Setting CSS variables on the root element restyles everything that uses them.
 */
function applyAppearance() {
    // Step 1: Initialize function scope and validate execution context
    const theme = themeById($('themeSelect').value);
    const layout = layoutById($('layoutSelect').value);
    const vars = theme.vars || {};
    const customAccent = $('customAccent').value || vars['--karl-accent'] || '#00c2ff';
    
    Object.entries(vars).forEach(([key, value]) => {
        document.documentElement.style.setProperty(key, value);
    });
    
    // The user's custom accent overrides the theme accent (applied after the theme vars on purpose).
    document.documentElement.style.setProperty('--karl-accent', customAccent);
    document.body.dataset.layout = layout.id;
    $('themeDescription').innerText = theme.description || '';
    $('layoutDescription').innerText = layout.description || '';

    // When syncing with VS Code, derive background/panel/border from the editor's own colours
    // (color-mix blends them with a dark base) so the panel matches the user's VS Code theme.
    // Presets variables modifications
    if ($('syncVsCodeTheme').checked) {
        document.documentElement.style.setProperty('--karl-bg', 'color-mix(in srgb, var(--vscode-sideBar-background) 82%, #020817)');
        document.documentElement.style.setProperty('--karl-panel', 'color-mix(in srgb, var(--vscode-editor-background) 88%, #07162a)');
        document.documentElement.style.setProperty('--karl-border', 'color-mix(in srgb, var(--vscode-widget-border) 60%, #0ea5ff)');
        document.documentElement.style.setProperty('--karl-text', 'var(--vscode-foreground)');
    }

    // High contrast and reduced motion are data attributes that CSS reacts to.
    if ($('highContrastMode').checked) {
        document.body.dataset.layoutContrast = 'high';
    } else {
        delete document.body.dataset.layoutContrast;
    }

    // Animation intensity is a percentage; 0 is treated the same as "reduced motion".
    const intensity = Number($('animationIntensity').value);
    document.documentElement.style.setProperty('--animation-intensity', intensity / 100);

    if ($('reducedMotion').checked || intensity === 0) {
        document.body.dataset.motion = 'reduced';
    } else {
        delete document.body.dataset.motion;
    }

    renderThemeCatalog();
    persist();
}

/**
 * Show one workspace tab (cockpit, chat, swarm, knowledge, lab, settings...) and hide the others.
 *
 * Some tabs lazily load their data on first view, and the main input of the tab gets focus.
 * @param {string} wsId
 */
function switchWorkspace(wsId) {
    // Step 1: Initialize function scope and validate execution context
    document.querySelectorAll('.workspace').forEach(section => section.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.workspace === wsId));
    const section = $(`workspace-${wsId}`);
    if (section) {
        section.classList.add('active');
    }

    // Lazy loading: fetch data only when its tab is opened to avoid needless backend calls.
    if (wsId === 'settings') {
        loadModels();
        loadAdapters();
    }
    if (wsId === 'knowledge') {
        loadKbSources();
        loadCodexTopics();
    }
    if (wsId === 'lab') loadPromptPairs();
    if (wsId === 'swarm' && isConnected()) loadAgentProfiles();
    
    // Focus Management
    if (wsId === 'chat') {
        const chatInput = $('chatInput');
        if (chatInput) chatInput.focus();
    } else if (wsId === 'swarm') {
        const objective = $('objective');
        if (objective) objective.focus();
    } else if (wsId === 'knowledge') {
        const kbQuery = $('kbQuery');
        if (kbQuery) kbQuery.focus();
    }

    // Ask the extension host for fresh cockpit state, then save which tab is open.
    // Request an update to sync cockpit variables
    vscode.postMessage({ command: 'get_cockpit_state' });
    persist();
}
