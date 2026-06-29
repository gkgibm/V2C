/**
 * V2C VS Code Extension — Main entry point
 *
 * Audio capture strategy:
 *   VS Code extensions run in Node.js and have NO access to the Web Audio API
 *   or getUserMedia. The solution is a hidden WebviewPanel — webviews run in a
 *   Chromium renderer context and DO have full Web Audio API access.
 *
 *   Flow:
 *     1. On "Start Listening", open a hidden WebviewPanel.
 *     2. The webview HTML calls getUserMedia + AudioWorklet to capture mic PCM.
 *     3. Each 100ms chunk of Float32 PCM is posted to the extension host via
 *        acquireVsCodeApi().postMessage({ type: 'AUDIO_CHUNK', data: [...] }).
 *     4. The extension host converts Float32 → Int16, base64-encodes it, and
 *        sends it to the Python server as AUDIO_CHUNK WebSocket messages.
 *     5. On "Stop Listening", the webview stops capture and the extension sends
 *        AUDIO_STOP, then closes the panel.
 *     6. The Python server transcribes, refines, classifies, and returns an
 *        ACTION message. The extension applies it as a WorkspaceEdit.
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
let audioPanel: vscode.WebviewPanel | null = null;
let extensionContext: vscode.ExtensionContext | null = null;

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
  extensionContext = context;

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
    _setStatus(connected ? "idle" : "disconnected");
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
  _destroyAudioPanel();
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
      "[V2C] Not connected to the Python server. Start it with: v2c-server"
    );
    return;
  }
  if (isListening) { return; }

  isListening = true;
  _setStatus("listening");

  // Send current editor context so the server can extract AST identifiers
  _sendEditorContext();

  // Open the hidden webview that captures mic audio
  _openAudioPanel();
}

function _stopListening(): void {
  if (!isListening) { return; }
  isListening = false;

  // Tell the webview to stop recording
  audioPanel?.webview.postMessage({ type: "STOP_RECORDING" });

  // Tell the Python server recording has ended → trigger transcription
  bridge?.sendAudioStop();
  _setStatus("processing");

  // Close the webview after a short delay (give it time to flush)
  setTimeout(_destroyAudioPanel, 300);
}

function _showStatus(): void {
  const connected = bridge?.isConnected ? "✅ Connected" : "❌ Disconnected";
  const listening = isListening ? "🎤 Listening" : "⏸ Idle";
  vscode.window.showInformationMessage(`[V2C] ${connected} | ${listening}`);
}

// ── Audio webview ─────────────────────────────────────────────────────────────

function _openAudioPanel(): void {
  if (audioPanel) { return; }

  audioPanel = vscode.window.createWebviewPanel(
    "v2cAudio",
    "V2C Audio",
    { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    {
      enableScripts: true,
      // No local resource roots needed — all inline HTML
      retainContextWhenHidden: true,
    }
  );

  // Make the panel invisible — move it off-screen via CSS
  audioPanel.webview.html = _getAudioWebviewHtml();

  // Receive PCM chunks from the webview
  audioPanel.webview.onDidReceiveMessage((msg: { type: string; data?: number[]; error?: string }) => {
    if (msg.type === "AUDIO_CHUNK" && msg.data) {
      // msg.data is Float32 samples as a plain number[] array
      // Convert Float32 → Int16 → base64 and send to server
      const float32 = new Float32Array(msg.data);
      const int16 = _float32ToInt16(float32);
      const b64 = Buffer.from(int16.buffer).toString("base64");
      bridge?.sendAudioChunk(b64);

    } else if (msg.type === "RECORDING_STARTED") {
      // Webview confirmed mic access granted
      console.log("[V2C] Microphone capture started");

    } else if (msg.type === "ERROR") {
      vscode.window.showErrorMessage(`[V2C] Microphone error: ${msg.error}`);
      _stopListening();
    }
  });

  audioPanel.onDidDispose(() => {
    audioPanel = null;
  });
}

function _destroyAudioPanel(): void {
  audioPanel?.dispose();
  audioPanel = null;
}

/** Convert Float32Array ([-1,1]) → Int16Array */
function _float32ToInt16(float32: Float32Array): Int16Array {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

/**
 * The webview HTML page.
 *
 * Uses getUserMedia + ScriptProcessor (wide browser support inside VS Code's
 * embedded Chromium) to capture mono 16 kHz PCM and post 100ms chunks back
 * to the extension host.
 *
 * AudioWorklet would be cleaner but requires a separate JS file served from
 * a local resource URI. ScriptProcessorNode works inline and is sufficient
 * for our latency requirements.
 */
function _getAudioWebviewHtml(): string {
  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {
      background: #1e1e1e;
      color: #ccc;
      font-family: monospace;
      font-size: 12px;
      padding: 8px;
      margin: 0;
    }
    #status { color: #4ec9b0; margin-bottom: 4px; }
    #error  { color: #f44; }
  </style>
</head>
<body>
  <div id="status">V2C — requesting microphone access…</div>
  <div id="error"></div>

  <script>
    const vscode = acquireVsCodeApi();
    const SAMPLE_RATE = 16000;
    const CHUNK_MS    = 100;          // send a chunk every 100 ms
    const BUFFER_SIZE = 4096;         // ScriptProcessorNode buffer (~256ms @16kHz)

    let audioCtx    = null;
    let source      = null;
    let processor   = null;
    let stream      = null;
    let accumulator = [];             // Float32 samples accumulated between sends
    let lastSend    = Date.now();
    let running     = false;

    // ── Start recording as soon as the page loads ──────────────────────────

    async function startRecording() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });

        // Create an AudioContext at exactly 16 kHz
        audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        source   = audioCtx.createMediaStreamSource(stream);

        // ScriptProcessorNode: fires onaudioprocess every BUFFER_SIZE samples
        processor = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);

        processor.onaudioprocess = (e) => {
          if (!running) return;
          // Copy the input channel data (Float32, range [-1, 1])
          const channelData = e.inputBuffer.getChannelData(0);
          for (let i = 0; i < channelData.length; i++) {
            accumulator.push(channelData[i]);
          }

          // Flush every CHUNK_MS milliseconds
          const now = Date.now();
          if (now - lastSend >= CHUNK_MS) {
            lastSend = now;
            if (accumulator.length > 0) {
              vscode.postMessage({ type: 'AUDIO_CHUNK', data: Array.from(accumulator) });
              accumulator = [];
            }
          }
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);
        running = true;

        document.getElementById('status').textContent = '🎤 Recording…';
        vscode.postMessage({ type: 'RECORDING_STARTED' });

      } catch (err) {
        document.getElementById('status').textContent = 'Error';
        document.getElementById('error').textContent  = err.message;
        vscode.postMessage({ type: 'ERROR', error: err.message });
      }
    }

    // ── Stop recording on message from extension host ──────────────────────

    window.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'STOP_RECORDING') {
        running = false;

        // Flush remaining samples
        if (accumulator.length > 0) {
          vscode.postMessage({ type: 'AUDIO_CHUNK', data: Array.from(accumulator) });
          accumulator = [];
        }

        // Tear down audio graph
        processor?.disconnect();
        source?.disconnect();
        stream?.getTracks().forEach(t => t.stop());
        audioCtx?.close();
        document.getElementById('status').textContent = '⏹ Stopped';
      }
    });

    startRecording();
  </script>
</body>
</html>`;
}

// ── Editor context helpers ────────────────────────────────────────────────────

function _sendEditorContext(): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor) { return; }
  const doc = editor.document;
  bridge?.sendContext({
    active_file: doc.fileName,
    language: doc.languageId,
    source_code: doc.getText().slice(0, 60_000),
    cursor_line: editor.selection.active.line,
    cursor_char: editor.selection.active.character,
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
  const endOfLine = new vscode.Position(
    line,
    editor.document.lineAt(line).range.end.character
  );
  const range = new vscode.Range(endOfLine, endOfLine);

  editor.setDecorations(transcriptDecorationType, [{
    range,
    renderOptions: { after: { contentText: ` ⟵ "${msg.refined}"` } },
  }]);

  // Clear after 4 seconds
  setTimeout(() => {
    editor.setDecorations(transcriptDecorationType, []);
  }, 4_000);
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
      const text = (action["text"] as string) ?? "";
      await editor.edit((eb) => eb.insert(editor.selection.active, text));
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
      vscode.window.showInformationMessage(
        `[V2C] Searching for '${name}' to delete — confirm in the results panel.`
      );
      await vscode.commands.executeCommand("workbench.action.findInFiles", {
        query: `def ${name}`,
        isRegex: false,
      });
      break;
    }

    case ACTION.NAVIGATE: {
      const navTarget = (action["target"] as string) ?? "";
      const name      = (action["name"]   as string) ?? "";
      if (navTarget === "LINE") {
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
      const description = (action["description"] as string) ?? "";
      await editor.insertSnippet(
        new vscode.SnippetString(`# TODO (voice): ${description}\n\${1:}`)
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
  idle:         { text: "$(mic) V2C",             tooltip: "V2C: Ready — Cmd+Shift+V to start",         warning: false },
  listening:    { text: "$(mic-filled) V2C Listening…", tooltip: "V2C: Listening — Cmd+Shift+V to stop",  warning: true  },
  processing:   { text: "$(loading~spin) V2C…",   tooltip: "V2C: Processing your command",               warning: false },
  disconnected: { text: "$(debug-disconnect) V2C (offline)", tooltip: "V2C: Server not running — start with: v2c-server", warning: false },
};

function _setStatus(kind: StatusKind): void {
  if (!statusBarItem) { return; }
  const cfg = STATUS_CONFIG[kind];
  statusBarItem.text = cfg.text;
  statusBarItem.tooltip = cfg.tooltip;
  statusBarItem.backgroundColor = cfg.warning
    ? new vscode.ThemeColor("statusBarItem.warningBackground")
    : undefined;
}
