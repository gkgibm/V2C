# V2C – Voice-to-Code

> **Privacy-first, AST-aware voice coding engine for Python — Phase 1 MVP**

V2C converts spoken developer intent into structured Python code edits. It runs
entirely on-device (no audio ever leaves your machine) and understands your
*codebase* rather than just your words.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            VS Code Extension (TypeScript)                 │
│   microphone toggle ▸ WebSocket client ▸ apply WorkspaceEdit / diff      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ WebSocket (localhost:6789)
┌────────────────────────────────▼─────────────────────────────────────────┐
│                        V2C Python Backend                                  │
│                                                                            │
│  ┌─────────────┐   raw PCM    ┌────────────┐  transcript  ┌────────────┐ │
│  │ AudioCapture│─────────────▶│ ASR Engine │─────────────▶│  Refiner   │ │
│  │ (sounddevice│              │ (Whisper)  │              │ (LLM/rules)│ │
│  │  + VAD)     │              └────────────┘              └─────┬──────┘ │
│  └─────────────┘                                                │        │
│                                                           refined text    │
│  ┌─────────────────────────────────────────────────────────────▼──────┐  │
│  │                     Intent Router                                    │  │
│  │   ┌──────────────┐          ┌───────────────────────────────────┐   │  │
│  │   │ Fast rule-   │ dictation│         LLM Command Parser         │   │  │
│  │   │ based mapper │◀─────────│  (context-aware code generation)   │   │  │
│  │   └──────┬───────┘  command └────────────────┬──────────────────┘   │  │
│  └──────────┼──────────────────────────────────┼────────────────────────┘  │
│             │                                   │                           │
│  ┌──────────▼───────────────────────────────────▼──────────────────────┐  │
│  │                  AST Engine  (tree-sitter Python)                    │  │
│  │  parse ▸ query ▸ validate ▸ produce WorkspaceEdit JSON               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Local-first ASR** | Audio never leaves the machine; low-latency |
| **Tree-sitter for Python AST** | Incremental, error-tolerant parsing; <5 ms queries |
| **Post-ASR LLM refinement** | Fixes phonetic drift, CamelCase splitting, symbol loss |
| **Hybrid intent router** | Fast rule-path for simple commands; LLM path for semantic edits |
| **WebSocket bridge** | Decouples Python engine from VS Code extension host |

---

## Project layout

```
V2C/
├── src/
│   └── v2c/
│       ├── __init__.py
│       ├── cli.py                  # Click CLI entry-point
│       ├── config.py               # Pydantic settings
│       ├── asr/
│       │   ├── __init__.py
│       │   ├── capture.py          # Microphone capture + VAD
│       │   ├── engine.py           # Whisper wrapper
│       │   └── refiner.py          # Post-ASR LLM refinement
│       ├── intent/
│       │   ├── __init__.py
│       │   ├── router.py           # Dictation vs. command classifier
│       │   ├── rules.py            # Deterministic rule-based mapper
│       │   └── llm_parser.py       # LLM-based semantic command parser
│       ├── ast_engine/
│       │   ├── __init__.py
│       │   ├── parser.py           # Tree-sitter Python parser
│       │   ├── queries.py          # Query builder
│       │   └── editor_action.py    # WorkspaceEdit data model
│       ├── bridge/
│       │   ├── __init__.py
│       │   ├── server.py           # asyncio WebSocket server
│       │   └── protocol.py         # JSON message schema (Pydantic)
│       └── queries/
│           └── python.scm          # Tree-sitter query file
├── extension/                      # VS Code TypeScript extension
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── extension.ts
│       └── bridge.ts
├── tests/
│   ├── conftest.py
│   ├── test_asr_engine.py
│   ├── test_refiner.py
│   ├── test_intent_router.py
│   ├── test_ast_engine.py
│   └── test_bridge_protocol.py
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd V2C

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install the package in editable mode (with dev extras)
pip install -e ".[dev]"

# 4. Copy and edit the environment file
cp .env.example .env
# Set OPENAI_API_KEY if you want cloud-based ASR refinement
# or leave it unset to run fully offline (Whisper tiny)

# 5. Start the WebSocket server (default port 6789)
v2c-server

# 6. Install the VS Code extension
cd extension && npm install && npm run compile
# Then press F5 in VS Code to launch the Extension Development Host
```

### Running tests

```bash
pytest
```

---

## Configuration

All settings are loaded via [`src/v2c/config.py`](src/v2c/config.py) from
environment variables (or `.env`).

| Variable | Default | Description |
|---|---|---|
| `V2C_ASR_MODEL` | `tiny` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `V2C_ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `V2C_VAD_AGGRESSIVENESS` | `2` | WebRTC VAD level (0–3) |
| `V2C_SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `V2C_WS_PORT` | `6789` | WebSocket bridge port |
| `V2C_REFINE_ENABLED` | `true` | Enable post-ASR LLM refinement |
| `V2C_OPENAI_API_KEY` | _(none)_ | OpenAI key (optional; falls back to rule-based) |
| `V2C_LLM_MODEL` | `gpt-4o-mini` | LLM model for refinement & command parsing |

---

## License

MIT © V2C Contributors
