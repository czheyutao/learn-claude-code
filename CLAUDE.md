# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a teaching repository for **harness engineering** — building the infrastructure (tools, context management, permissions, etc.) that surrounds an AI model to make it an effective agent. It uses Claude Code's architecture as the reference implementation. The model provides intelligence; the harness provides the action space.

12 progressive sessions (s01–s12) each add one harness mechanism on top of the same core agent loop. `s_full.py` combines all mechanisms into a complete reference implementation.

## Commands

### Python agents

```sh
pip install -r requirements.txt       # anthropic, python-dotenv, pyyaml
cp .env.example .env                  # set ANTHROPIC_API_KEY and MODEL_ID
python agents/s01_agent_loop.py       # simplest agent loop
python agents/s_full.py               # complete capstone
```

### Web platform (Next.js 16 + Tailwind CSS v4)

```sh
cd web && npm ci && npm run dev       # http://localhost:3000
```

Note: `predev`/`prebuild` scripts run `tsx scripts/extract-content.ts` to extract docs content into the app — this must succeed before the dev server or build will start.

### Tests and CI

```sh
cd web && npx tsc --noEmit            # TypeScript type-check
cd web && npm run build               # production build
python -m pytest tests/test_agents_smoke.py -q   # compile-check all agent scripts
python -m pytest tests/test_s_full_background.py -q  # unit tests for BackgroundManager
```

## Architecture

```
agents/          # Python: 13 self-contained, runnable scripts (s01–s12 + s_full)
docs/{en,zh,ja}/ # Mental-model-first documentation (3 languages)
web/             # Next.js interactive learning platform
skills/          # SKILL.md files for s05 skill-loading mechanism
tests/           # pytest smoke tests
```

### The core agent loop (unchanging across all sessions)

```python
while True:
    response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        return
    # execute each tool_use block, append tool_result
    results = [execute_tool(block) for block in response.content if block.type == "tool_use"]
    messages.append({"role": "user", "content": results})
```

Each session layers one harness mechanism on top — without changing the loop.

### Session progression

| Phase | Sessions | Mechanisms |
|-------|----------|------------|
| 1: The Loop | s01, s02 | agent loop, tool dispatch via `name → handler` map |
| 2: Planning & Knowledge | s03–s06 | TodoWrite, subagents (fresh messages[] per child), on-demand skill loading, 3-layer context compaction |
| 3: Persistence | s07, s08 | file-based task CRUD + dependency graph, background daemon threads with notification queue |
| 4: Teams | s09–s12 | persistent teammates + JSONL mailboxes, shutdown/plan-approval protocols, autonomous idle cycle + auto-claim, worktree isolation |

### Key environment variables

- `ANTHROPIC_API_KEY` — required for API access
- `ANTHROPIC_BASE_URL` — optional; when set, `ANTHROPIC_AUTH_TOKEN` is cleared before client init
- `MODEL_ID` — the Anthropic model to use (e.g., `claude-sonnet-4-6`)

### Agent file conventions

- Every agent file in `agents/` is self-contained and runnable directly: `python agents/sXX_name.py`
- Files follow a pattern: imports → config → tool definitions/implementations → helper functions → `main()` or interactive REPL
- s01 is minimal (one tool, ~120 lines); s_full.py is comprehensive (~1000 lines, all tools)
- `s12_worktree_task_isolation.py` includes a minimal append-only lifecycle event stream for teaching (not production-grade)

### Web architecture

- Next.js 16 App Router with TypeScript
- Content extraction: `scripts/extract-content.ts` parses docs/ into structured JSON consumed by the app
- `src/components/` — UI components (diagrams, code viewers, step-through visualizations)
- `src/data/` — extracted content cache
- `src/i18n/` — internationalization for en/zh/ja
- `src/lib/` — utility functions, source code loading
- `src/hooks/` — React hooks
- `src/types/` — TypeScript type definitions

### Test patterns

- `test_agents_smoke.py` — parametrized test that `py_compile`s every `agents/*.py` file to catch syntax errors
- `test_s_full_background.py` — imports `s_full.py` via `importlib` with mocked `anthropic`/`dotenv`, tests isolated background task logic

### Documentation style

Docs follow a "mental-model-first" pattern: problem statement → solution → ASCII diagram → minimal code. This applies across all 3 languages in `docs/`.
