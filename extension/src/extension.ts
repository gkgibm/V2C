/**
 * V2C VS Code Extension — Main entry point
 *
 * Audio capture model:
 *   getUserMedia is unavailable in VS Code webviews (blocked on non-https
 *   origins). Instead, the extension sends START_RECORDING / STOP_RECORDING
 *   commands over the WebSocket. The Python server (which has sounddevice
 *   already installed) captures the microphone locally and runs the full
 *   ASR → refine → classify → act pipeline.
 *
 *   Flow:
 *     Cmd+Shift+V (start)
 *       → send CONTEXT (editor state for AST identifier extraction)
 *       → send START_RECORDING
 *       → server opens mic, status bar turns yellow
 *     Cmd+Shift+V (stop)
 *       → send STOP_RECORDING
 *       → server closes mic, transcribes, classifies
 *       → server sends TRANSCRIPT + ACTION
 *       → extension applies ACTION as WorkspaceEdit
 */

import * as vscode from "vscode";
import { V2CBridge, ActionMessage, TranscriptMessage, StatusMessage } from "./bridge";

// ── Action type constants (must match editor_action.py) ──────────────────────

const ACTION = {
  DICTATION: "DICTATION",
  ADD_FUNCTION: "ADD_FUNCTION",
  ADD_CLASS: "ADD_CLASS",
  ADD_METHOD: "ADD_METHOD",
  DELETE_FUNCTION: "DELETE_FUNCTION",
  DELETE_CLASS: "DELETE_CLASS",
  ADD_IMPORT: "ADD_IMPORT",
  RENAME: "RENAME",
  NAVIGATE: "NAVIGATE",
  GENERATE: "GENERATE",
} as const;

// ── Extension state ───────────────────────────────────────────────────────────

let bridge: V2CBridge | null = null;
let statusBarItem: vscode.StatusBarItem | null = null;
let isListening = false;

// Decoration type for showing the refined transcript inline
const transcriptDecorationType = vscode.window.createTextEditorDecorationType({
  after: {
    color: new vscode.ThemeColor("editorCodeLens.foreground"),
    fontStyle: "italic",
    margin: "0 0 0 2em",
  },
});

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
  bridge.onTranscript.event(_handleTranscript);
  bridge.onAction.event(_handleAction);
  bridge.onServerError.event((msg) => {
    vscode.window.showErrorMessage(`[V2C] Server error: ${msg.message}`);
  });
  bridge.onConnectionChange.event((connected) => {
    _setStatus(connected ? "idle" : "disconnected");
    // If the server died mid-recording, reset state
    if (!connected && isListening) { isListening = false; }
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
  transcriptDecorationType.dispose();
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
  _setStatus("listening");

  // 1. Send editor context so the server can extract AST identifiers
  _sendEditorContext();

  // 2. Ask the Python server to open its microphone
  bridge.sendStartRecording();
}

function _stopListening(): void {
  if (!isListening) { return; }
  isListening = false;
  _setStatus("processing");

  // Ask the Python server to stop the mic and run the pipeline
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

function _handleTranscript(msg: TranscriptMessage): void {
  const showTranscript = vscode.workspace
    .getConfiguration("v2c")
    .get<boolean>("showTranscript", true);
  if (!showTranscript) { return; }

  const editor = vscode.window.activeTextEditor;
  if (!editor) { return; }

  const line = editor.selection.active.line;
  const eol  = new vscode.Position(line, editor.document.lineAt(line).range.end.character);

  editor.setDecorations(transcriptDecorationType, [{
    range: new vscode.Range(eol, eol),
    renderOptions: { after: { contentText: ` ⟵ "${msg.refined}"` } },
  }]);

  setTimeout(() => editor.setDecorations(transcriptDecorationType, []), 4_000);
}

async function _handleAction(msg: ActionMessage): Promise<void> {
  const { action, action_id } = msg;
  const actionType = (action["action_type"] as string) ?? "";

  try {
    await _applyAction(actionType, action);
    bridge?.sendAck(action_id);
  } catch (err) {
    vscode.window.showErrorMessage(`[V2C] Failed to apply action: ${err}`);
  } finally {
    _setStatus("idle");
  }
}

// ── Action executor ───────────────────────────────────────────────────────────

async function _applyAction(
  actionType: string,
  action: Record<string, unknown>
): Promise<void> {
  const editor = vscode.window.activeTextEditor;

  switch (actionType) {
    case ACTION.DICTATION: {
      if (!editor) { return; }
      await editor.edit((eb) =>
        eb.insert(editor.selection.active, (action["text"] as string) ?? "")
      );
      break;
    }

    case ACTION.ADD_FUNCTION: {
      if (!editor) { return; }
      const name   = (action["target_name"] as string) ?? "new_function";
      const params = ((action["parameters"] as string[]) ?? []).join(", ");
      await editor.insertSnippet(
        new vscode.SnippetString(`\ndef ${name}(${params}):\n    \${1:pass}\n`)
      );
      break;
    }

    case ACTION.ADD_CLASS: {
      if (!editor) { return; }
      const name = (action["target_name"] as string) ?? "NewClass";
      await editor.insertSnippet(
        new vscode.SnippetString(
          `\nclass ${name}:\n    def __init__(self) -> None:\n        \${1:pass}\n`
        )
      );
      break;
    }

    case ACTION.ADD_IMPORT: {
      if (!editor) { return; }
      const module = (action["target_name"] as string) ?? "";
      if (!module) { return; }
      await editor.edit((eb) =>
        eb.insert(new vscode.Position(0, 0), `import ${module}\n`)
      );
      break;
    }

    case ACTION.DELETE_FUNCTION:
    case ACTION.DELETE_CLASS: {
      const name = (action["target_name"] as string) ?? "";
      if (!name || !editor) { return; }
      await vscode.commands.executeCommand("workbench.action.findInFiles", {
        query: `def ${name}`, isRegex: false,
      });
      break;
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
      break;
    }

    case ACTION.GENERATE: {
      if (!editor) { return; }
      await editor.insertSnippet(
        new vscode.SnippetString(
          `# TODO (voice): ${(action["description"] as string) ?? ""}\n\${1:}`
        )
      );
      break;
    }

    default:
      console.warn(`[V2C] Unknown action type: ${actionType}`);
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
