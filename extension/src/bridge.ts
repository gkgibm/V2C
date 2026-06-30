/**
 * V2C VS Code Extension — WebSocket Bridge
 *
 * Manages the persistent WebSocket connection to the Python V2C server
 * and exposes a clean API for sending messages and receiving typed
 * server messages.
 *
 * Architecture:
 *   - Connects (with exponential back-off retry) to ws://127.0.0.1:6789
 *   - Sends context snapshots before every recording
 *   - Streams raw PCM audio chunks from the Web Audio API
 *   - Receives STATUS, TRANSCRIPT, and ACTION messages
 *   - Emits typed events that extension.ts subscribes to
 */

import * as vscode from "vscode";
import WebSocket = require("ws");

// ── Message type constants (must match v2c/bridge/protocol.py) ────────────────

export const MSG = {
  // Client → Server
  CONTEXT: "CONTEXT",
  START_RECORDING: "START_RECORDING",
  STOP_RECORDING: "STOP_RECORDING",
  AUDIO_CHUNK: "AUDIO_CHUNK",
  AUDIO_STOP: "AUDIO_STOP",
  ACK: "ACK",

  // Server → Client
  STATUS: "STATUS",
  PARTIAL_TRANSCRIPT: "PARTIAL_TRANSCRIPT",
  TRANSCRIPT: "TRANSCRIPT",
  ACTION: "ACTION",
  SERVER_ERROR: "SERVER_ERROR",
} as const;

// ── Shared types ───────────────────────────────────────────────────────────────

export interface EditorContext {
  active_file: string;
  language: string;
  source_code: string;
  cursor_line: number;
  cursor_char: number;
  selected_text: string;
}

export interface StatusMessage {
  type: "STATUS";
  status: "READY" | "LISTENING" | "PROCESSING" | "IDLE";
  detail: string;
}

export interface TranscriptMessage {
  type: "TRANSCRIPT";
  raw: string;
  refined: string;
}

export interface ActionMessage {
  type: "ACTION";
  action_id: string;
  action: Record<string, unknown>;
}

export interface PartialTranscriptMessage {
  type: "PARTIAL_TRANSCRIPT";
  text: string;
  is_final: boolean;
}

export interface ServerErrorMessage {
  type: "SERVER_ERROR";
  message: string;
}

export type ServerMessage =
  | StatusMessage
  | PartialTranscriptMessage
  | TranscriptMessage
  | ActionMessage
  | ServerErrorMessage;

// ── Retry configuration ────────────────────────────────────────────────────────

const INITIAL_RETRY_MS = 1_000;
const MAX_RETRY_MS = 30_000;
const BACKOFF_FACTOR = 2;

// ── Bridge class ──────────────────────────────────────────────────────────────

export class V2CBridge {
  private ws: WebSocket | null = null;
  private retryDelay: number = INITIAL_RETRY_MS;
  private retryTimer: NodeJS.Timeout | null = null;
  private disposed: boolean = false;

  // Event emitters — extension.ts subscribes to these
  readonly onStatus = new vscode.EventEmitter<StatusMessage>();
  readonly onPartialTranscript = new vscode.EventEmitter<PartialTranscriptMessage>();
  readonly onTranscript = new vscode.EventEmitter<TranscriptMessage>();
  readonly onAction = new vscode.EventEmitter<ActionMessage>();
  readonly onServerError = new vscode.EventEmitter<ServerErrorMessage>();
  readonly onConnectionChange = new vscode.EventEmitter<boolean>();

  constructor(private readonly config: () => { host: string; port: number }) {}

  // ── Connection management ────────────────────────────────────────────────

  connect(): void {
    if (this.disposed) return;
    const { host, port } = this.config();
    const url = `ws://${host}:${port}`;

    this.ws = new WebSocket(url);

    this.ws.on("open", () => {
      this.retryDelay = INITIAL_RETRY_MS;
      this.onConnectionChange.fire(true);
    });

    this.ws.on("message", (data: WebSocket.RawData) => {
      try {
        const msg: ServerMessage = JSON.parse(data.toString());
        this._dispatch(msg);
      } catch (err) {
        console.error("[V2C] Failed to parse server message", err);
      }
    });

    this.ws.on("close", () => {
      this.onConnectionChange.fire(false);
      this._scheduleReconnect();
    });

    this.ws.on("error", (err: Error) => {
      console.error(`[V2C] WebSocket error: ${err.message}`);
    });
  }

  disconnect(): void {
    this._clearRetryTimer();
    this.ws?.close();
    this.ws = null;
  }

  dispose(): void {
    this.disposed = true;
    this.disconnect();
    this.onStatus.dispose();
    this.onPartialTranscript.dispose();
    this.onTranscript.dispose();
    this.onAction.dispose();
    this.onServerError.dispose();
    this.onConnectionChange.dispose();
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ── Sending helpers ──────────────────────────────────────────────────────

  sendContext(ctx: EditorContext): void {
    this._send({ type: MSG.CONTEXT, context: ctx });
  }

  sendStartRecording(): void {
    this._send({ type: MSG.START_RECORDING });
  }

  sendStopRecording(): void {
    this._send({ type: MSG.STOP_RECORDING });
  }

  sendAudioChunk(pcmBase64: string): void {
    this._send({ type: MSG.AUDIO_CHUNK, data_b64: pcmBase64 });
  }

  sendAudioStop(): void {
    this._send({ type: MSG.AUDIO_STOP });
  }

  sendAck(actionId: string): void {
    this._send({ type: MSG.ACK, action_id: actionId });
  }

  private _send(payload: Record<string, unknown>): void {
    if (!this.isConnected) {
      console.warn("[V2C] Attempted to send while disconnected");
      return;
    }
    this.ws!.send(JSON.stringify(payload));
  }

  // ── Dispatch ─────────────────────────────────────────────────────────────

  private _dispatch(msg: ServerMessage): void {
    switch (msg.type) {
      case MSG.STATUS:
        this.onStatus.fire(msg as StatusMessage);
        break;
      case MSG.PARTIAL_TRANSCRIPT:
        this.onPartialTranscript.fire(msg as PartialTranscriptMessage);
        break;
      case MSG.TRANSCRIPT:
        this.onTranscript.fire(msg as TranscriptMessage);
        break;
      case MSG.ACTION:
        this.onAction.fire(msg as ActionMessage);
        break;
      case MSG.SERVER_ERROR:
        this.onServerError.fire(msg as ServerErrorMessage);
        break;
    }
  }

  // ── Reconnect logic ──────────────────────────────────────────────────────

  private _scheduleReconnect(): void {
    if (this.disposed) return;
    this._clearRetryTimer();
    this.retryTimer = setTimeout(() => {
      this.retryDelay = Math.min(this.retryDelay * BACKOFF_FACTOR, MAX_RETRY_MS);
      this.connect();
    }, this.retryDelay);
  }

  private _clearRetryTimer(): void {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }
}
