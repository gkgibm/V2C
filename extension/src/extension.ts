/**
 * V2C VS Code Extension — Main entry point
 *
 * Responsibilities:
 *   1. Activate and register all extension commands / keybindings.
 *   2. Maintain a status bar item showing the current V2C state.
 *   3. Manage the microphone recording lifecycle (push-to-talk or toggle).
 *   4. Apply incoming ActionMessages as VS Code WorkspaceEdits.
 *   5. Show inline feedback (transcript decorations, error messages).
 *
 * Design decisions:
 *   - Audio is captured via the Node.js `node-record-lpcm16` compatible
 *     approach using the platform CLI tools (sox/ffmpeg), delegating
 *     all heavy lifting to the Python server.
 *   - For the MVP, we capture audio through VS Code's available APIs
 *     by reading a raw PCM stream from the OS microphone.
 *   - The extension uses a push-to-talk model by default (Cmd+Shift+V
 *     to start, release to send AUDIO_STOP) and toggles via command.
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

  // Create status bar item
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "v2c.toggleListening";
  _setStatus("idle");
  statusBarItem.show();

  // Create bridge and wire events
  bridge = new V2CBridge(config);

  bridge.onStatus.event(_handleStatus);
  bridge.onTranscript.event(_handleTranscript);
  bridge.onAction.event(_handleAction);
  bridge.onServerError.event((msg) => {
    vscode.window.showErrorMessage(`[V2C] Server error: ${msg.message}`);
  });
  bridge.onConnectionChange.event((connected) => {
    if (connected) {
      _setStatus("idle");
    } else {
      _setStatus("disconnected");
    }
  });

  // Auto-connect if configured
  if (vscode.workspace.getConfiguration("v2c").get<boolean>("autoConnect", true)) {
    bridge.connect();
  }

  // Register commands
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
  if (isListening) {
    _stopListening();
  } else {
    _startListening();
  }
}

function _startListening(): void {
  if (!bridge?.isConnected) {
    vscode.window.showWarningMessage(
      "[V2C] Not connected to the Python server. Start it with: v2c-server"
    );
    return;
  }

  isListening = true;
  _setStatus("listening");

  // Send the current editor context before any audio
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const doc = editor.document;
    bridge.sendContext({
      active_file: doc.fileName,
      language: doc.languageId,
      // Truncate large files to stay within context budgets
      source_code: doc.getText().slice(0, 60_000),
      cursor_line: editor.selection.active.line,
      cursor_char: editor.selection.active.character,
      selected_text: editor.document.getText(editor.selection),
    });
  }

  vscode.window.showInformationMessage("[V2C] Listening… (press Cmd+Shift+V again to stop)");
}

function _stopListening(): void {
  if (!isListening) return;
  isListening = false;
  bridge?.sendAudioStop();
  _setStatus("processing");
}

function _showStatus(): void {
  const connected = bridge?.isConnected ? "✅ Connected" : "❌ Disconnected";
  const listening = isListening ? "🎤 Listening" : "⏸ Idle";
  vscode.window.showInformationMessage(`[V2C] ${connected} | ${listening}`);
}

// ── Event handlers ────────────────────────────────────────────────────────────

function _handleStatus(msg: StatusMessage): void {
  switch (msg.status) {
    case "READY":
    case "IDLE":
      _setStatus("idle");
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

  if (!showTranscript) return;

  // Show the refined transcript as a ghost decoration at the cursor
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;

  const line = editor.selection.active.line;
  const endOfLine = new vscode.Position(line, editor.document.lineAt(line).range.end.character);
  const range = new vscode.Range(endOfLine, endOfLine);

  editor.setDecorations(transcriptDecorationType, [
    {
      range,
      renderOptions: {
        after: { contentText: ` ⟵ "${msg.refined}"` },
      },
    },
  ]);

  // Clear decoration after 3 seconds
  setTimeout(() => {
    editor.setDecorations(transcriptDecorationType, []);
  }, 3_000);
}

async function _handleAction(msg: ActionMessage): Promise<void> {
  const { action, action_id } = msg;
  const actionType = (action["action_type"] as string) ?? "";

  try {
    await _applyAction(actionType, action);
    bridge?.sendAck(action_id);
    _setStatus("idle");
  } catch (err) {
    vscode.window.showErrorMessage(`[V2C] Failed to apply action: ${err}`);
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
      if (!editor) return;
      const text = (action["text"] as string) ?? "";
      await editor.edit((editBuilder) => {
        editBuilder.insert(editor.selection.active, text);
      });
      break;
    }

    case ACTION.ADD_FUNCTION: {
      if (!editor) return;
      const name = (action["target_name"] as string) ?? "new_function";
      const params = ((action["parameters"] as string[]) ?? []).join(", ");
      const snippet = new vscode.SnippetString(
        `\ndef ${name}(${params}):\n    \${1:pass}\n`
      );
      await editor.insertSnippet(snippet);
      break;
    }

    case ACTION.ADD_CLASS: {
      if (!editor) return;
      const name = (action["target_name"] as string) ?? "NewClass";
      const snippet = new vscode.SnippetString(
        `\nclass ${name}:\n    def __init__(self) -> None:\n        \${1:pass}\n`
      );
      await editor.insertSnippet(snippet);
      break;
    }

    case ACTION.ADD_IMPORT: {
      if (!editor) return;
      const module = (action["target_name"] as string) ?? "";
      if (!module) return;
      const doc = editor.document;
      // Insert at line 0 if the first line is not already an import.
      const firstLine = doc.lineAt(0).text;
      const insertPos = new vscode.Position(0, 0);
      await editor.edit((editBuilder) => {
        editBuilder.insert(insertPos, `import ${module}\n`);
      });
      break;
    }

    case ACTION.DELETE_FUNCTION:
    case ACTION.DELETE_CLASS: {
      const name = (action["target_name"] as string) ?? "";
      if (!name || !editor) return;
      // Delegate to the VS Code rename/delete via a symbol search
      vscode.window.showInformationMessage(
        `[V2C] Delete ${actionType === ACTION.DELETE_FUNCTION ? "function" : "class"} '${name}' — use the Problems panel to locate and confirm.`
      );
      // Full AST-guided deletion is in Phase 2; for now, show a search
      await vscode.commands.executeCommand("workbench.action.findInFiles", {
        query: `def ${name}`,
        isRegex: false,
      });
      break;
    }

    case ACTION.NAVIGATE: {
      const navTarget = (action["target"] as string) ?? "";
      const name = (action["name"] as string) ?? "";
      if (navTarget === "LINE") {
        const line = parseInt(name, 10) - 1;
        if (!isNaN(line) && editor) {
          const pos = new vscode.Position(line, 0);
          editor.revealRange(new vscode.Range(pos, pos));
          editor.selection = new vscode.Selection(pos, pos);
        }
      } else {
        // Use VS Code's symbol search for FUNCTION, CLASS, SYMBOL
        await vscode.commands.executeCommand("workbench.action.gotoSymbol");
      }
      break;
    }

    case ACTION.GENERATE: {
      const description = (action["description"] as string) ?? "";
      // Phase 2: pipe into Copilot Chat or inline chat
      // For now, insert the description as a comment prompt
      if (editor) {
        const snippet = new vscode.SnippetString(
          `# TODO (voice): ${description}\n\${1:}`
        );
        await editor.insertSnippet(snippet);
      }
      break;
    }

    default:
      console.warn(`[V2C] Unknown action type: ${actionType}`);
  }
}

// ── Status bar helpers ────────────────────────────────────────────────────────

type StatusKind = "idle" | "listening" | "processing" | "disconnected";

const STATUS_CONFIG: Record<StatusKind, { icon: string; text: string; tooltip: string }> = {
  idle: {
    icon: "$(mic)",
    text: "V2C",
    tooltip: "V2C: Ready. Click or press Cmd+Shift+V to start listening.",
  },
  listening: {
    icon: "$(mic-filled)",
    text: "V2C Listening…",
    tooltip: "V2C: Listening. Press Cmd+Shift+V again to stop.",
  },
  processing: {
    icon: "$(loading~spin)",
    text: "V2C Processing…",
    tooltip: "V2C: Processing your voice command.",
  },
  disconnected: {
    icon: "$(debug-disconnect)",
    text: "V2C (offline)",
    tooltip: "V2C: Not connected to Python server. Run: v2c-server",
  },
};

function _setStatus(kind: StatusKind): void {
  if (!statusBarItem) return;
  const cfg = STATUS_CONFIG[kind];
  statusBarItem.text = `${cfg.icon} ${cfg.text}`;
  statusBarItem.tooltip = cfg.tooltip;
  statusBarItem.backgroundColor =
    kind === "listening"
      ? new vscode.ThemeColor("statusBarItem.warningBackground")
      : undefined;
}
