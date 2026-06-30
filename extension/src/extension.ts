/**
 * V2C VS Code Extension — Main entry point
 *
 * Live-code model:
 *   As the user speaks, each partial ASR transcript is run through the fast
 *   rule-based pipeline on the server. The result arrives as a
 *   LIVE_ACTION(is_partial=true) message. The extension applies it to the
 *   document immediately — real code appears as you speak.
 *
 *   When the next partial arrives the previous partial edit is undone first,
 *   then the updated code is inserted. On stop, the server sends a final
 *   LIVE_ACTION(is_partial=false) from the full ASR+refine pipeline. The
 *   extension undoes the last partial and applies the final version once.
 *
 *   Flow:
 *     Cmd+Shift+V (start)
 *       → send CONTEXT (editor state)
 *       → send START_RECORDING
 *       → every ~1s: LIVE_ACTION(partial) → undo prev → apply code live
 *     Cmd+Shift+V (stop)
 *       → send STOP_RECORDING
 *       → LIVE_ACTION(final) → undo last partial → apply final clean code
 */

import * as vscode from "vscode";
import { V2CBridge, ActionMessage, TranscriptMessage, StatusMessage, LiveActionMessage } from "./bridge";

// ── Action type constants (must match editor_action.py) ──────────────────────

const ACTION = {
  DICTATION:       "DICTATION",
  ADD_FUNCTION:    "ADD_FUNCTION",
  ADD_CLASS:       "ADD_CLASS",
  ADD_METHOD:      "ADD_METHOD",
  DELETE_FUNCTION: "DELETE_FUNCTION",
  DELETE_CLASS:    "DELETE_CLASS",
  ADD_IMPORT:      "ADD_IMPORT",
  RENAME:          "RENAME",
  NAVIGATE:        "NAVIGATE",
  GENERATE:        "GENERATE",
  NEWLINE:         "NEWLINE",
} as const;

// ── Extension state ───────────────────────────────────────────────────────────

let bridge: V2CBridge | null = null;
let statusBarItem: vscode.StatusBarItem | null = null;
let isListening = false;

// Live-edit state: how many undos are needed to remove the last partial edit.
// Structural snippets (function/class) count as 1 undo step; dictation also 1.
let _pendingUndoSteps = 0;

// ── Activation ────────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  const config = () => ({
    host: vscode.workspace.getConfiguration("v2c").get<string>("serverHost", "127.0.0.1"),
    port: vscode.workspace.getConfiguration("v2c").get<number>("serverPort", 6789),
  });

  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "v2c.toggleListening";
  _setStatus("idle");
  statusBarItem.show();

  // Bridge
  bridge = new V2CBridge(config);
  bridge.onStatus.event(_handleStatus);
  bridge.onLiveAction.event(_handleLiveAction);
  bridge.onTranscript.event(_handleTranscript);
  bridge.onAction.event(_handleAction);
  bridge.onServerError.event((msg) => {
    vscode.window.showErrorMessage(`[V2C] Server error: ${msg.message}`);
  });
  bridge.onConnectionChange.event((connected) => {
    _setStatus(connected ? "idle" : "disconnected");
    if (!connected && isListening) {
      isListening = false;
      _pendingUndoSteps = 0;
    }
  });

  if (vscode.workspace.getConfiguration("v2c").get<boolean>("autoConnect", true)) {
    bridge.connect();
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("v2c.startListening", _startListening),
    vscode.commands.registerCommand("v2c.stopListening", _stopListening),
    vscode.commands.registerCommand("v2c.toggleListening", _toggleListening),
    vscode.commands.registerCommand("v2c.showStatus", _showStatus),
    statusBarItem,
    bridge,
  );
}

export function deactivate(): void {
  bridge?.dispose();
}

// ── Commands ──────────────────────────────────────────────────────────────────

function _toggleListening(): void {
  isListening ? _stopListening() : _startListening();
}

function _startListening(): void {
  if (!bridge?.isConnected) {
    vscode.window.showWarningMessage(
      "[V2C] Not connected — start the server first:  v2c-server"
    );
    return;
  }
  if (isListening) { return; }

  isListening = true;
  _pendingUndoSteps = 0;
  _setStatus("listening");

  _sendEditorContext();
  bridge.sendStartRecording();
}

function _stopListening(): void {
  if (!isListening) { return; }
  isListening = false;
  _setStatus("processing");
  bridge?.sendStopRecording();
}

function _showStatus(): void {
  const connected = bridge?.isConnected ? "✅ Connected" : "❌ Disconnected";
  const state     = isListening         ? "🎤 Listening" : "⏸ Idle";
  vscode.window.showInformationMessage(`[V2C] ${connected} | ${state}`);
}

// ── Editor context ────────────────────────────────────────────────────────────

function _sendEditorContext(): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor) { return; }
  const doc = editor.document;
  bridge?.sendContext({
    active_file:   doc.fileName,
    language:      doc.languageId,
    source_code:   doc.getText().slice(0, 60_000),
    cursor_line:   editor.selection.active.line,
    cursor_char:   editor.selection.active.character,
    selected_text: doc.getText(editor.selection),
  });
}

// ── Event handlers ────────────────────────────────────────────────────────────

function _handleStatus(msg: StatusMessage): void {
  switch (msg.status) {
    case "READY":
    case "IDLE":
      if (!isListening) { _setStatus("idle"); }
      break;
    case "LISTENING":
      _setStatus("listening");
      break;
    case "PROCESSING":
      _setStatus("processing");
      break;
  }
}

async function _handleLiveAction(msg: LiveActionMessage): Promise<void> {
  // Step 1 — undo the previous partial edit so we start from a clean slate.
  if (_pendingUndoSteps > 0) {
    for (let i = 0; i < _pendingUndoSteps; i++) {
      await vscode.commands.executeCommand("undo");
    }
    _pendingUndoSteps = 0;
  }

  if (msg.is_partial) {
    // Step 2 — apply the new partial code into the document.
    const steps = await _applyAction(msg.action["action_type"] as string, msg.action);
    _pendingUndoSteps = steps;
    // Don't ack partial actions — server doesn't wait for it.
  } else {
    // Final action: apply and keep. Reset undo state.
    await _applyAction(msg.action["action_type"] as string, msg.action);
    _pendingUndoSteps = 0;
    bridge?.sendAck(msg.action_id);
    _setStatus("idle");
  }
}

// Legacy ACTION message handler (kept for compatibility, same logic)
async function _handleAction(msg: ActionMessage): Promise<void> {
  const { action, action_id } = msg;
  try {
    await _applyAction(action["action_type"] as string, action);
    bridge?.sendAck(action_id);
  } catch (err) {
    vscode.window.showErrorMessage(`[V2C] Failed to apply action: ${err}`);
  } finally {
    _setStatus("idle");
  }
}

function _handleTranscript(_msg: TranscriptMessage): void {
  // Transcript is informational only in live-edit mode.
  // No decoration shown — the code is already in the document.
}

// ── Action executor — returns number of undo steps applied ───────────────────

async function _applyAction(
  actionType: string,
  action: Record<string, unknown>
): Promise<number> {
  const editor = vscode.window.activeTextEditor;

  switch (actionType) {
    case ACTION.DICTATION: {
      if (!editor) { return 0; }
      const text = (action["text"] as string) ?? "";
      if (!text) { return 0; }
      await editor.edit((eb) => eb.insert(editor.selection.active, text));
      return 1;
    }

    case ACTION.ADD_FUNCTION: {
      if (!editor) { return 0; }
      const name   = (action["target_name"] as string) ?? "new_function";
      const params = ((action["parameters"] as string[]) ?? []).join(", ");
      await editor.insertSnippet(
        new vscode.SnippetString(`\ndef ${name}(${params}):\n    \${1:pass}\n`)
      );
      return 1;
    }

    case ACTION.ADD_CLASS: {
      if (!editor) { return 0; }
      const name = (action["target_name"] as string) ?? "NewClass";
      await editor.insertSnippet(
        new vscode.SnippetString(
          `\nclass ${name}:\n    def __init__(self) -> None:\n        \${1:pass}\n`
        )
      );
      return 1;
    }

    case ACTION.ADD_IMPORT: {
      if (!editor) { return 0; }
      const module = (action["target_name"] as string) ?? "";
      if (!module) { return 0; }
      await editor.edit((eb) =>
        eb.insert(new vscode.Position(0, 0), `import ${module}\n`)
      );
      return 1;
    }

    case ACTION.DELETE_FUNCTION:
    case ACTION.DELETE_CLASS: {
      const name = (action["target_name"] as string) ?? "";
      if (!name || !editor) { return 0; }
      await vscode.commands.executeCommand("workbench.action.findInFiles", {
        query: `def ${name}`, isRegex: false,
      });
      return 0; // no document edit to undo
    }

    case ACTION.NAVIGATE: {
      const target = (action["target"] as string) ?? "";
      const name   = (action["name"]   as string) ?? "";
      if (target === "LINE") {
        const line = parseInt(name, 10) - 1;
        if (!isNaN(line) && editor) {
          const pos = new vscode.Position(Math.max(0, line), 0);
          editor.revealRange(new vscode.Range(pos, pos));
          editor.selection = new vscode.Selection(pos, pos);
        }
      } else {
        await vscode.commands.executeCommand("workbench.action.gotoSymbol");
      }
      return 0; // navigation has no document edit to undo
    }

    case ACTION.GENERATE: {
      if (!editor) { return 0; }
      await editor.insertSnippet(
        new vscode.SnippetString(
          `# TODO (voice): ${(action["description"] as string) ?? ""}\n\${1:}`
        )
      );
      return 1;
    }

    case ACTION.NEWLINE: {
      if (!editor) { return 0; }
      const count = (action["count"] as number) ?? 1;
      for (let i = 0; i < count; i++) {
        await vscode.commands.executeCommand("editor.action.insertLineAfter");
      }
      return count;
    }

    default:
      console.warn(`[V2C] Unknown action type: ${actionType}`);
      return 0;
  }
}

// ── Status bar ────────────────────────────────────────────────────────────────

type StatusKind = "idle" | "listening" | "processing" | "disconnected";

const STATUS_CONFIG: Record<StatusKind, { text: string; tooltip: string; warning: boolean }> = {
  idle:         { text: "$(mic) V2C",                   tooltip: "V2C: Ready — Cmd+Shift+V to start",         warning: false },
  listening:    { text: "$(mic-filled) V2C Listening…", tooltip: "V2C: Listening — Cmd+Shift+V to stop",      warning: true  },
  processing:   { text: "$(loading~spin) V2C…",         tooltip: "V2C: Processing your command",              warning: false },
  disconnected: { text: "$(debug-disconnect) V2C",      tooltip: "V2C: Server offline — run: v2c-server",     warning: false },
};

function _setStatus(kind: StatusKind): void {
  if (!statusBarItem) { return; }
  const cfg = STATUS_CONFIG[kind];
  statusBarItem.text        = cfg.text;
  statusBarItem.tooltip     = cfg.tooltip;
  statusBarItem.backgroundColor = cfg.warning
    ? new vscode.ThemeColor("statusBarItem.warningBackground")
    : undefined;
}
