# ANBV Wajo proactive email agent

This repository implements an actual agent loop for proactive email management:

`observe -> normalize -> assess risk -> interpret -> choose autonomy -> act or wait -> learn`

The LLM is the agent's semantic planner, but it has no mailbox tools. Deterministic Python policy
owns authority, approvals bind exact payloads, and learning can reduce interruptions only for safe,
reversible actions.

## Quick start

```powershell
uv sync --extra dev
uv run wajo demo --reset
uv run pytest
```

Use the OpenAI planner only when `OPENAI_API_KEY` is configured:

```powershell
uv run wajo process data/fixtures/newsletter.json --planner openai
```

The default demo uses a deterministic offline planner so reviewers can reproduce the complete
agent lifecycle without credentials or network access.

Useful commands:

```powershell
uv run wajo --db wajo.db ingest data/fixtures/newsletter.json
uv run wajo --db wajo.db inbox
uv run wajo --db wajo.db decision <decision-id>
uv run wajo --db wajo.db approve <approval-id>
uv run wajo --db wajo.db preferences
uv run wajo --db wajo.db events
uv run wajo --db wajo.db --json inbox
```

The `--db`, `--mailbox`, `--actor`, `--model`, and `--json` options are global, so they appear
before the command name. All mailbox effects in the offline demo use the scripted mock adapter.
Human approval displays the exact bound payload and asks for confirmation. JSON automation must
make consent explicit with `approve <approval-id> --yes`.

See `DESIGN.md` for the concise technical design and `anbv_wajo.txt` for the exhaustive
interviewer walkthrough.
