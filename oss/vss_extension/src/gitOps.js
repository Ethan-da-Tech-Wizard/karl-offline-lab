// @ts-check
// ==============================================================================
// Program: gitOps.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Git integration utility extracting branch states, changed files, and repository metadata.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// Node built-in used to spawn the git executable; no third-party git library is needed.
const cp = require('child_process');

/**
 * Executes a git command.
 * @param {string[]} args
 * @param {string} cwd
 * @returns {Promise<string>}
 */
function execGit(args, cwd) {
    // Wrap the callback-style execFile API in a Promise so callers can use async/await.
    return new Promise((resolve, reject) => {
        // execFile (not exec) passes args as an array with no shell, so branch names or paths
        // cannot inject shell commands. The 8 MB maxBuffer allows large diffs without a stream error.
        cp.execFile('git', args, { cwd, maxBuffer: 1024 * 1024 * 8 }, (err, stdout, stderr) => {
            if (err) {
                // Prefer git's own error text; fall back to the Node error message.
                reject(new Error(stderr || err.message));
            } else {
                resolve(stdout);
            }
        });
    });
}

/**
 * Gets the current git branch name.
 * @param {string} workspacePath
 * @returns {Promise<string>}
 */
async function getGitBranch(workspacePath) {
    // No workspace folder open means there is no repository to ask.
    if (!workspacePath) return '';
    try {
        // "--show-current" prints an empty line on a detached HEAD, which becomes '' here.
        const branch = await execGit(['branch', '--show-current'], workspacePath);
        return branch.trim();
    } catch {
        // Not a git repo, git not installed or any other failure: report "no branch" instead of throwing.
        return '';
    }
}

// Public API of this module.
module.exports = {
    execGit,
    getGitBranch
};
