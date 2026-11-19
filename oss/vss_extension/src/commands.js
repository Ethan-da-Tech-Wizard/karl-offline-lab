// @ts-check
// ==============================================================================
// Program: commands.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Command registry and dispatcher handling user invocations, diff actions, and RAG transfers.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// vscode is supplied by the editor at runtime. packageContext bounds payload size (fileOps.js)
// and execGit runs git without a shell (gitOps.js).
const vscode = require('vscode');
const path = require('path');
const { packageContext } = require('./fileOps');
const { execGit } = require('./gitOps');

/**
 * Catalogue of one-click Karl workflows, keyed by id.
 *
 * Each entry describes: a title and description for menus; precondition flags
 * (``requiresSelection``, ``requiresFile``, ``requiresGit``) that buildWorkflowContext() enforces;
 * ``targetTab`` (which Karl tab opens: chat, swarm, git, diagnostics, knowledge or vision); and
 * ``promptBuilder(context)``, which turns the gathered {filepath, code, objective} into the
 * prompt text. extension.js registers each id as a "karl.<id>" command.
 * @type {Record<string, any>}
 */
const WORKFLOW_REGISTRY = {
    // Code-selection workflows (open the chat tab).
    fixSelection: {
        // Despite the id, the title and prompt say "Refactor": the id is kept stable because
        // package.json commands and saved keybindings refer to it.
        id: 'fixSelection',
        title: 'Refactor Selection',
        description: 'Refactor code selection for clarity, safety, and maintainability.',
        requiresSelection: true,
        requiresFile: true,
        targetTab: 'chat',
        promptBuilder: (context) => `Refactor this code from ${context.filepath}:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    explainSelection: {
        id: 'explainSelection',
        title: 'Explain Selection',
        description: 'Explain selection control flow, risks, and intent.',
        requiresSelection: true,
        requiresFile: true,
        targetTab: 'chat',
        promptBuilder: (context) => `Explain this code from ${context.filepath}:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    // File-level workflows (open the swarm tab): they send the whole active file.
    generateTests: {
        // Whole-file workflows go to the swarm tab because they may touch more than one file,
        // unlike the selection workflows which stay in single-turn chat.
        id: 'generateTests',
        title: 'Generate Tests',
        description: 'Generate unit tests for the active file.',
        requiresFile: true,
        targetTab: 'swarm',
        promptBuilder: (context) => `Generate unit tests for the active file: ${context.filepath}\n\nCode:\n${context.code}\n\nObjective: ${context.objective}`
    },
    reviewActiveFile: {
        id: 'reviewActiveFile',
        title: 'Review Active File',
        description: 'Review the active file for bugs, regressions, and improvements.',
        requiresFile: true,
        targetTab: 'swarm',
        promptBuilder: (context) => `Review this file: ${context.filepath}\n\nCode:\n${context.code}\n\nObjective: ${context.objective}`
    },
    sendCurrentFileToSwarm: {
        id: 'sendCurrentFileToSwarm',
        title: 'Send File to Swarm',
        description: 'Send current file and objective to Karl Swarm.',
        requiresFile: true,
        targetTab: 'swarm',
        promptBuilder: (context) => `Implement the changes in: ${context.filepath}\n\nCode:\n${context.code}\n\nObjective: ${context.objective}`
    },
    // Git workflows: ``code`` is filled with git output instead of editor text.
    reviewStagedDiff: {
        // The three diff-review workflows differ only in the git command run in buildWorkflowContext()
        // (--staged, plain, HEAD); the prompt text just labels which diff the model is looking at.
        id: 'reviewStagedDiff',
        title: 'Review Staged Diff',
        description: 'Review the staged git diff.',
        requiresGit: true,
        targetTab: 'git',
        promptBuilder: (context) => `Review this staged git diff:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    reviewUnstagedDiff: {
        id: 'reviewUnstagedDiff',
        title: 'Review Unstaged Diff',
        description: 'Review the unstaged git diff.',
        requiresGit: true,
        targetTab: 'git',
        promptBuilder: (context) => `Review this unstaged git diff:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    reviewCombinedDiff: {
        id: 'reviewCombinedDiff',
        title: 'Review Combined Diff',
        description: 'Review all staged and unstaged changes.',
        requiresGit: true,
        targetTab: 'git',
        promptBuilder: (context) => `Review the combined git diff (all modified files):\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    generateCommitMessage: {
        // Reuses the staged diff on purpose: a commit message must describe exactly what will be
        // committed, not unstaged edits.
        id: 'generateCommitMessage',
        title: 'Generate Commit Message',
        description: 'Generate a conventional commit message from staged changes.',
        requiresGit: true,
        targetTab: 'git',
        promptBuilder: (context) => `Generate a conventional commit message from this staged diff:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    branchSummary: {
        // Has no diff to review, so buildWorkflowContext() assembles branch/status/log text instead and
        // skips the "empty diff" check.
        id: 'branchSummary',
        title: 'Summarize Git Branch',
        description: 'Summarize the current branch status and commits.',
        requiresGit: true,
        targetTab: 'git',
        promptBuilder: (context) => `Summarize current git branch state:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    // Diagnostics workflows: ``code`` is a JSON summary of the editor's problems.
    explainDiagnostics: {
        // No requiresFile flag: workspace-wide diagnostics are valid with no editor open.
        id: 'explainDiagnostics',
        title: 'Explain Diagnostics',
        description: 'Explain all workspace diagnostics.',
        targetTab: 'diagnostics',
        promptBuilder: (context) => `Explain these workspace diagnostics:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    explainCurrentFileDiagnostics: {
        id: 'explainCurrentFileDiagnostics',
        title: 'Explain File Diagnostics',
        description: 'Explain active file diagnostics.',
        requiresFile: true,
        targetTab: 'diagnostics',
        promptBuilder: (context) => `Explain these diagnostics in active file:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    // Free-form workflows with no code payload, only the user's objective.
    askWorkspace: {
        id: 'askWorkspace',
        title: 'Ask About Workspace',
        description: 'Ask Karl about the workspace architecture.',
        targetTab: 'chat',
        promptBuilder: (context) => `Question about the workspace:\n\nObjective: ${context.objective}`
    },
    createImplementationPlan: {
        // No requiresFile/requiresSelection flag, so buildWorkflowContext() gathers no code for this
        // workflow: ${context.code} below is empty and only the user's objective carries the request,
        // despite the "selected files" wording.
        id: 'createImplementationPlan',
        title: 'Create Implementation Plan',
        description: 'Create an implementation plan from selected files.',
        targetTab: 'swarm',
        promptBuilder: (context) => `Create an implementation plan across the selected files:\n\n${context.code}\n\nObjective: ${context.objective}`
    },
    // Special case: runWorkflow() sends the selection to knowledge-base search instead of a chat prompt.
    searchKbSelection: {
        // Needs both flags because the query text is the selection; runWorkflow() special-cases this id
        // so nothing is sent to the model.
        id: 'searchKbSelection',
        title: 'Search KB',
        description: 'Search the Knowledge Base for selected text.',
        requiresSelection: true,
        requiresFile: true,
        targetTab: 'knowledge',
        promptBuilder: (context) => `${context.code}`
    },
    // Vision workflows: ``filepath`` is an image file chosen by the user.
    analyzeImage: {
        // The prompt references the image by path: the webview/engine reads the file itself, so no
        // image bytes travel through this extension.
        id: 'analyzeImage',
        title: 'Analyze Image',
        description: 'Analyze an image using Karl Vision.',
        targetTab: 'vision',
        promptBuilder: (context) => `Analyze this image at ${context.filepath}\n\nObjective: ${context.objective}`
    },
    reviewScreenshotError: {
        id: 'reviewScreenshotError',
        title: 'Review Screenshot Error',
        description: 'Review screenshot of an error.',
        targetTab: 'vision',
        promptBuilder: (context) => `Review this error screenshot at ${context.filepath}\n\nObjective: ${context.objective}`
    }
};

/**
 * Returns the current workspace path or fallback folder.
 * @param {string} [fallbackFile]
 * @returns {string}
 */
function currentWorkspacePath(fallbackFile) {
    // The first workspace folder is treated as the project root.
    if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
        return vscode.workspace.workspaceFolders[0].uri.fsPath;
    }
    return fallbackFile ? path.dirname(fallbackFile) : '';
}

/**
 * Aggregates diagnostic counts and files details.
 * @param {boolean} [currentFileOnly]
 * @returns {{counts: {error: number, warning: number, info: number, hint: number}, files: Object<string, any[]>}}
 */
function groupedDiagnostics(currentFileOnly = false) {
    // Remember the active file so currentFileOnly can filter to it.
    // Computed once, outside the loop, because getDiagnostics() can return thousands of entries.
    const activeFile = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.document.uri.fsPath : '';
    // A Map keeps files in the order VS Code reports them; converted to a plain object at the end
    // so the result serialises with JSON.stringify.
    const groups = new Map();
    const counts = { error: 0, warning: 0, info: 0, hint: 0 };
    // Index = VS Code DiagnosticSeverity (0 Error, 1 Warning, 2 Information, 3 Hint).
    const severityName = ['error', 'warning', 'info', 'hint'];
    for (const [uri, items] of vscode.languages.getDiagnostics()) {
        if (currentFileOnly && uri.fsPath !== activeFile) continue;
        // Cap at 25 diagnostics per file; a few hundred lint messages would flood the prompt.
        const fileItems = items.slice(0, 25).map(item => {
            const severity = severityName[item.severity] || 'info';
            counts[severity] += 1;
            return {
                severity,
                message: item.message,
                source: item.source || '',
                // VS Code positions are 0-based; people (and the model) expect 1-based lines and columns.
                line: item.range.start.line + 1,
                character: item.range.start.character + 1
            };
        });
        if (fileItems.length) groups.set(uri.fsPath, fileItems);
    }
    return { counts, files: Object.fromEntries(groups) };
}

/**
 * Focuses the Karl panel and waits for ready status.
 * @param {any} sidebarProvider
 * @returns {Promise<void>}
 */
async function revealKarlPanel(sidebarProvider) {
    // The id is the container id declared in package.json (viewsContainers); executing it opens
    // the activity-bar view and creates the webview if it was never shown.
    await vscode.commands.executeCommand('workbench.view.extension.karl-swarm');
    // The webview may not exist yet; wait until it reports ready before posting messages to it.
    await sidebarProvider.waitForWebview();
}

/**
 * Runs a specified workflow structure.
 * @param {any} sidebarProvider
 * @param {any} workflow
 * @param {vscode.Uri | null} [customUri]
 * @returns {Promise<void>}
 */
async function runWorkflow(sidebarProvider, workflow, customUri = null) {
    // A null context means a precondition failed and the user was already told why.
    const context = await buildWorkflowContext(workflow, sidebarProvider, customUri);
    if (!context) return;

    // finalCode starts as the full text and may be replaced by a bounded head/tail version below.
    let finalCode = context.code;
    let isTruncated = false;
    let isSummary = false;
    const originalChars = finalCode.length;

    // Large payloads: let the user choose between the full text and a head/tail summary.
    // 30k characters is the point where the prompt is likely to crowd out a small local model's
    // context; the hard 70k cap lives in packageContext().
    if (originalChars > 30000) {
        const selection = await vscode.window.showWarningMessage(
            `The packaged context is large (${originalChars} characters). How would you like to proceed?`,
            'Send Full Context',
            'Send Bounded Summary (Head/Tail)',
            'Cancel'
        );

        if (selection === 'Cancel' || !selection) {
            return;
        }

        if (selection === 'Send Bounded Summary (Head/Tail)') {
            isSummary = true;
        }
    }

    // Final size enforcement; this may truncate even when the user picked "Send Full Context" (70k limit).
    const packaged = packageContext(finalCode, workflow.title, isSummary);
    finalCode = packaged.code;
    isTruncated = packaged.meta.truncated;
    isSummary = packaged.meta.summaryOnly;

    await revealKarlPanel(sidebarProvider);

    // Knowledge-base search needs just the text, not a workflow start.
    if (workflow.id === 'searchKbSelection') {
        sidebarProvider.postMessageToWebview({
            command: 'search_kb_text',
            query: finalCode
        });
        return;
    }

    // The webview receives everything it needs to start the workflow.
    sidebarProvider.postMessageToWebview({
        command: 'start_workflow',
        workflowId: workflow.id,
        data: {
            mode: workflow.title,
            objective: context.objective || workflow.description,
            code: finalCode,
            filepath: context.filepath || workflow.title,
            workspace_path: currentWorkspacePath(context.filepath),
            targetTab: workflow.targetTab,
            prompt: workflow.promptBuilder ? workflow.promptBuilder(context) : '',
            context_meta: {
                label: workflow.title,
                originalChars,
                sentChars: finalCode.length,
                truncated: isTruncated,
                summaryOnly: isSummary
            }
        }
    });
}

/**
 * Runs workflow command by registered ID.
 * @param {any} sidebarProvider
 * @param {string} workflowId
 * @param {any} [_payload]
 * @returns {Promise<void>}
 */
async function runWorkflowById(sidebarProvider, workflowId, _payload = {}) {
    // Unknown ids are ignored silently.
    // The registry is a plain object, so inherited keys like "constructor" would also be
    // found here; ids come from package.json commands, not user input, which keeps that harmless.
    const workflow = WORKFLOW_REGISTRY[workflowId];
    if (workflow) {
        await runWorkflow(sidebarProvider, workflow);
    }
}

/**
 * Assembles and prompts the context payload for workflows.
 * @param {any} workflow
 * @param {any} sidebarProvider
 * @param {vscode.Uri | null} [customUri]
 * @returns {Promise<{code: string, filepath: string, objective: string} | null>}
 */
async function buildWorkflowContext(workflow, sidebarProvider, customUri = null) {
    // Output fields built up below depending on which preconditions the workflow declares.
    let code = '';
    let filepath = '';
    let objective = '';

    const editor = vscode.window.activeTextEditor;

    // Selection workflows need highlighted text in the active editor.
    if (workflow.requiresSelection) {
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found.');
            return null;
        }
        // Only the selection (not the whole file) is sent so the model's attention stays on the
        // code the user actually highlighted.
        const selection = editor.selection;
        code = editor.document.getText(selection);
        if (!code.trim()) {
            vscode.window.showWarningMessage('Please highlight some code first.');
            return null;
        }
        filepath = editor.document.uri.fsPath;
    } else if (workflow.requiresFile) {
        // File workflows use the file from the explorer/context menu (customUri) if given,
        // otherwise the active editor.
        if (customUri) {
            filepath = customUri.fsPath;
            try {
                const stat = await vscode.workspace.fs.stat(customUri);
                // Only read regular files; for folders ``code`` stays empty.
                if (stat.type === vscode.FileType.File) {
                    const bytes = await vscode.workspace.fs.readFile(customUri);
                    code = new TextDecoder('utf-8').decode(bytes);
                }
            } catch (err) {
                vscode.window.showErrorMessage(`Failed to read file: ${err.message}`);
                return null;
            }
        } else {
            if (!editor) {
                vscode.window.showErrorMessage('No active text editor found.');
                return null;
            }
            filepath = editor.document.uri.fsPath;
            code = editor.document.getText();
        }
    }

    // Git workflows: run the appropriate git command in the workspace root.
    if (workflow.requiresGit) {
        const workspacePath = currentWorkspacePath('');
        if (!workspacePath) {
            vscode.window.showWarningMessage('Open a workspace before using git workflows.');
            return null;
        }
        try {
            // generateCommitMessage deliberately reuses the staged diff.
            if (workflow.id === 'reviewStagedDiff' || workflow.id === 'generateCommitMessage') {
                // execGit takes an argument array and runs git without a shell, so file names or branch names
                // cannot be interpreted as shell syntax.
                code = await execGit(['diff', '--staged'], workspacePath);
            } else if (workflow.id === 'reviewUnstagedDiff') {
                code = await execGit(['diff'], workspacePath);
            } else if (workflow.id === 'reviewCombinedDiff') {
                code = await execGit(['diff', 'HEAD'], workspacePath);
            } else if (workflow.id === 'branchSummary') {
                // Three read-only git queries run in parallel and are combined into one summary text.
                const [branch, status, recent] = await Promise.all([
                    execGit(['branch', '--show-current'], workspacePath),
                    execGit(['status', '--short', '--branch'], workspacePath),
                    execGit(['log', '--oneline', '--decorate', '-12'], workspacePath)
                ]);
                code = `Current branch: ${branch.trim() || '(detached)'}\n\nStatus:\n${status}\n\nRecent commits:\n${recent}`;
            }
        } catch (err) {
            vscode.window.showErrorMessage(`Git execution failed: ${err.message}`);
            return null;
        }

        // An empty diff means nothing to review (the branch summary is always non-empty).
        if (workflow.id !== 'branchSummary' && !code.trim()) {
            vscode.window.showInformationMessage('No git changes found.');
            return null;
        }
    }

    // Diagnostics workflows: gather problems from the editor, not from git or a file.
    if (workflow.id === 'explainDiagnostics' || workflow.id === 'explainCurrentFileDiagnostics') {
        const currentFileOnly = (workflow.id === 'explainCurrentFileDiagnostics');
        const diagnostics = groupedDiagnostics(currentFileOnly);
        if (!Object.keys(diagnostics.files).length) {
            vscode.window.showInformationMessage(currentFileOnly ? 'No diagnostics found for the active file.' : 'No current diagnostics found.');
            return null;
        }
        code = JSON.stringify(diagnostics, null, 2);
    }

    // Vision workflows: use the given/open file, else ask the user to pick an image.
    if (workflow.id === 'analyzeImage' || workflow.id === 'reviewScreenshotError') {
        let pathFromUri = customUri && customUri.fsPath;
        // Falling back to the active editor lets "Analyze Image" work on an image tab that is open
        // in VS Code's built-in image preview.
        if (!pathFromUri && editor) {
            pathFromUri = editor.document.uri.fsPath;
        }
        if (!pathFromUri) {
            const uris = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                canSelectFolders: false,
                canSelectMany: false,
                filters: {
                    'Images': ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp']
                },
                title: 'Select an image for Karl Vision'
            });
            if (uris && uris[0]) {
                pathFromUri = uris[0].fsPath;
            }
        }
        if (!pathFromUri) return null;
        filepath = pathFromUri;
        code = `Image File: ${filepath}`;
    }

    // Workflows in this list ask the user what to do and pre-fill a sensible default.
    const needsPrompt = [
        'fixSelection', 'explainSelection', 'generateTests', 'reviewActiveFile',
        // Git and diagnostics workflows are not in this list because their objective is fixed by
        // the prompt template; only workflows where the user's intent varies ask for input.
        'sendCurrentFileToSwarm', 'createImplementationPlan', 'askWorkspace',
        'analyzeImage', 'reviewScreenshotError'
    ].includes(workflow.id);

    if (needsPrompt) {
        let defaultValue = workflow.description;
        if (workflow.id === 'generateTests') defaultValue = 'Generate focused tests for the active file. Preserve existing behavior.';
        if (workflow.id === 'reviewActiveFile') defaultValue = 'Review the active file for bugs, regressions, missing tests, and concrete improvements.';
        if (workflow.id === 'sendCurrentFileToSwarm') defaultValue = 'Implement the requested change in this file and verify behavior.';
        if (workflow.id === 'analyzeImage') defaultValue = 'Describe this image, extract text (OCR), and identify any UI or visual issues.';
        if (workflow.id === 'reviewScreenshotError') defaultValue = 'Review this screenshot for error messages, stack traces, and propose fixes.';

        const userObjective = await vscode.window.showInputBox({
            prompt: `${workflow.title}: What should Karl do?`,
            value: defaultValue
        });
        // Pressing Escape cancels the workflow; an empty answer falls back to the default text.
        if (userObjective === undefined) return null;
        objective = userObjective.trim() || defaultValue;
    }

    return { code, filepath, objective };
}

/**
 * Relays the active editor file context to KB tab.
 * @param {any} sidebarProvider
 * @param {vscode.Uri | null} [uri]
 * @returns {Promise<void>}
 */
async function sendActiveFileToKb(sidebarProvider, uri) {
    // Explorer context menu supplies a URI; otherwise use the active editor's file.
    // `uri` is only passed by the explorer context menu; command-palette calls pass nothing.
    const selectedPath = uri && uri.fsPath ? uri.fsPath : null;
    const editor = vscode.window.activeTextEditor;
    const filePath = selectedPath || (editor ? editor.document.uri.fsPath : '');
    if (!filePath) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }
    await revealKarlPanel(sidebarProvider);
    sidebarProvider.postMessageToWebview({
        command: 'set_kb_path',
        path: filePath
    });
}

/**
 * Relays the workspace folder context to KB tab.
 * @param {any} sidebarProvider
 * @param {vscode.Uri | null} [uri]
 * @returns {Promise<void>}
 */
async function sendWorkspaceFolderToKb(sidebarProvider, uri) {
    // Prefer the folder right-clicked in the explorer, else the workspace root.
    // Same convention as sendActiveFileToKb: explorer menu passes a URI, palette does not.
    const workspacePath = uri && uri.fsPath ? uri.fsPath : currentWorkspacePath('');
    if (!workspacePath) {
        vscode.window.showWarningMessage('Open a workspace before ingesting a folder.');
        return;
    }
    await revealKarlPanel(sidebarProvider);
    sidebarProvider.postMessageToWebview({
        command: 'set_kb_path',
        path: workspacePath
    });
}

// Exported for extension.js, sidebarProvider.js and tests.
module.exports = {
    WORKFLOW_REGISTRY,
    currentWorkspacePath,
    groupedDiagnostics,
    revealKarlPanel,
    runWorkflow,
    runWorkflowById,
    buildWorkflowContext,
    sendActiveFileToKb,
    sendWorkspaceFolderToKb
};
