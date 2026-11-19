// ==============================================================================
// Program: run.js
// Programmer: Ethan-da-Tech-Wizard
// Last Modified: November 18, 2026
// Purpose: Automated test harness runner executing extension integration tests headlessly.
//          Coordinates editor events, client state, and backend RPC messaging.
// ==============================================================================

// Mock the 'vscode' module before requiring any file that depends on it
// Tests run in plain Node, where the "vscode" module does not exist. Patch require()
// so any file that imports it gets this small fake.
const Module = require('module');
const originalRequire = Module.prototype.require;
Module.prototype.require = function(id) {
    if (id === 'vscode') {
        return {
            Uri: {
                file: (path) => ({ fsPath: path })
            },
            commands: {
                executeCommand: async () => {}
            }
        };
    }
    return originalRequire.apply(this, arguments);
};

// Built-in assert: a failed check throws and makes the script exit non-zero.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

// Safe to require now that the vscode stub is installed.
const { packageContext } = require('../fileOps');

console.log('--- Starting Karl Extension Unit Tests ---');

// Small text must pass through unchanged; 80,000 characters must be truncated and summarized.
// 1. Test packageContext
console.log('Testing packageContext...');
const smallText = 'Hello World';
const packagedSmall = packageContext(smallText, 'TestLabel', false);
assert.strictEqual(packagedSmall.code, smallText);
assert.strictEqual(packagedSmall.meta.truncated, false);
assert.strictEqual(packagedSmall.meta.summaryOnly, false);

const largeText = 'A'.repeat(80000);
const packagedLarge = packageContext(largeText, 'TestLabel', false);
assert.strictEqual(packagedLarge.meta.truncated, true);
assert.strictEqual(packagedLarge.meta.summaryOnly, true);
assert.ok(packagedLarge.code.includes('middle omitted'));

// karl_render.js is a browser script, not a module, so it is loaded into a vm sandbox
// with stub globals; its functions then become properties of the sandbox object.
// 2. Test escapeHtml
console.log('Testing escapeHtml...');
const renderCode = fs.readFileSync(path.join(__dirname, '../../media/karl_render.js'), 'utf8');
// Minimal fake browser globals that karl_render.js touches while loading.
const sandbox = {
    $: () => {},
    document: {},
    window: {},
    lastKbSnapshot: {},
    kbSelectedSource: '',
    lastKbResults: [],
    pendingEdits: new Map(),
    conversationBranches: [],
    activeBranchId: '',
    responseThinkActive: false
};
vm.createContext(sandbox);
vm.runInContext(renderCode, sandbox);
const escapeHtml = sandbox.escapeHtml;

// Covers every escaped character: < > & " and the single quote.
assert.strictEqual(escapeHtml('<div>&"\'</div>'), '&lt;div&gt;&amp;&quot;&#39;&lt;/div&gt;');
assert.strictEqual(escapeHtml(''), '');
assert.strictEqual(escapeHtml(null), '');

console.log('Testing buildModelCard...');
const buildModelCard = sandbox.buildModelCard;
// The card should embed the title, meta text and the button markup unescaped.
const cardHtml = buildModelCard('MyModel', 'meta info', '<button>Btn</button>');
assert.ok(cardHtml.includes('MyModel'));
assert.ok(cardHtml.includes('meta info'));
assert.ok(cardHtml.includes('<button>Btn</button>'));

console.log('--- All Tests Passed! ---');
