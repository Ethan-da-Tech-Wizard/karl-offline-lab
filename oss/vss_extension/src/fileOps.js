// @ts-check
// ==============================================================================
// Program: fileOps.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Filesystem operations helper managing temporary files, diff buffers, and backup snapshots.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

const path = require('path');
const os = require('os');
const crypto = require('crypto');
const vscode = require('vscode');

// Size limits (in characters) for text sent to the backend: anything over 70,000 is
// replaced by a ~18,000 character head+tail summary so requests stay small.
const MAX_CONTEXT_CHARS = 70000;
const SUMMARY_CONTEXT_CHARS = 18000;

/**
 * Packages file or git context into a bounded payload.
 * @param {any} raw
 * @param {string} label
 * @param {boolean} [summaryOnly]
 * @returns {{code: string, meta: {label: string, originalChars: number, sentChars: number, truncated: boolean, summaryOnly: boolean}}}
 */
function packageContext(raw, label, summaryOnly = false) {
    // Coerce null/undefined/non-strings to a string so .length and .slice always work.
    const text = String(raw || '');
    const originalChars = text.length;
    // Fast path: small enough, send everything unchanged.
    if (originalChars <= MAX_CONTEXT_CHARS && !summaryOnly) {
        return {
            code: text,
            meta: { label, originalChars, sentChars: originalChars, truncated: false, summaryOnly: false }
        };
    }
    // Keep 65% of the budget from the start of the text and 35% from the end;
    // the beginning (imports, definitions) and the end (latest changes) are the most informative.
    const head = text.slice(0, Math.floor(SUMMARY_CONTEXT_CHARS * 0.65));
    const tail = text.slice(-Math.floor(SUMMARY_CONTEXT_CHARS * 0.35));
    // The notice tells the model the text was cut, so it asks for a narrower selection instead of guessing.
    const notice = [
        `[Karl context notice: ${label} was ${originalChars} characters and exceeded the safe extension payload threshold of ${MAX_CONTEXT_CHARS}.`,
        `The extension sent a bounded head/tail summary of ${head.length + tail.length} characters. Ask for a narrower file, selection, or diff if full precision is needed.]`,
        ''
    ].join('\n');
    return {
        code: `${notice}${head}\n\n[Karl context notice: middle omitted]\n\n${tail}`,
        meta: { label, originalChars, sentChars: notice.length + head.length + tail.length, truncated: true, summaryOnly: true }
    };
}

/**
 * Writes content to a temp file and opens a vscode diff.
 * @param {string} filename
 * @param {string} targetPath
 * @param {string} content
 * @param {string} title
 * @returns {Promise<string>}
 */
async function writeTempFileAndDiff(filename, targetPath, content, title) {
    // A random directory name per call avoids collisions between concurrent diffs.
    const randomHex = crypto.randomBytes(8).toString('hex');
    const tempProposedDir = path.join(os.tmpdir(), `karl-proposed-${randomHex}`);
    const proposedPath = path.join(tempProposedDir, filename);
    const proposedUri = vscode.Uri.file(proposedPath);

    // Use the VS Code fs API (works for remote workspaces too) to create the temp folder.
    await vscode.workspace.fs.createDirectory(vscode.Uri.file(tempProposedDir));
    const encoder = new TextEncoder();
    await vscode.workspace.fs.writeFile(proposedUri, encoder.encode(content));

    // Open VS Code's built-in diff editor: current file on the left, proposed content on the right.
    await vscode.commands.executeCommand(
        'vscode.diff',
        vscode.Uri.file(targetPath),
        proposedUri,
        title
    );
    // Callers use this path to apply or clean up the proposal later.
    return proposedPath;
}

module.exports = {
    MAX_CONTEXT_CHARS,
    SUMMARY_CONTEXT_CHARS,
    packageContext,
    writeTempFileAndDiff
};
