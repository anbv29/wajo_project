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

See `DESIGN.md` for the concise technical design and `anbv_wajo.txt` for the exhaustive
interviewer walkthrough.
