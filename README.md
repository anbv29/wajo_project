# ANBV Wajo proactive email agent

This repository implements an actual agent loop for proactive email management:

`observe -> normalize -> assess risk -> interpret -> choose autonomy -> act or wait -> learn`

The LLM is the agent's semantic planner, but it has no mailbox tools. Deterministic Python policy
owns authority, approvals bind exact payloads, and learning can reduce interruptions only for safe,
reversible actions.

## Quick start

```powershell
uv sync --extra dev --extra eval
uv run wajo demo --reset
uv run pytest
uv run python scripts/run_quality_gates.py
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

Optional Gmail observation is documented in `GMAIL_SETUP.md`. It uses a dedicated test account,
an injected short-lived access token, and dry-run mutations by default:

```powershell
uv run wajo --db gmail-test.db gmail-ingest <gmail-message-id>
```

The frozen synthetic evaluation pack can be validated without a model key or mailbox account:

```powershell
uv run python scripts/check_datasets.py
uv run python scripts/run_evaluation.py
uv run python scripts/run_failure_evaluation.py
```

See `data/evaluation/README.md` for its splits, labels, governance, and limitations. Regeneration is
an explicit maintenance action (`scripts/build_eval_datasets.py --force`), not part of a normal
evaluation run.

Development is the default evaluation split. Scoring frozen held-out cases requires deliberate
confirmation, which prevents an ordinary local check from turning the final set into tuning data:

```powershell
uv run python scripts/run_evaluation.py --split held_out --confirm-held-out
```

Add `--planner openai --repeats 3` for an explicitly networked variability run after configuring
`OPENAI_API_KEY`; offline evaluation is deterministic and credential-free.

Generate the complete submission report from development data with:

```powershell
uv run python scripts/generate_reports.py
```

This writes versioned raw JSON, reproducibility metadata, predeclared release gates, a readable
summary, 95% Wilson confidence intervals, and three PNG charts to `reports/`. Held-out report
generation is separately guarded by `--confirm-held-out`, just like direct evaluation.

See `DESIGN.md` for the concise technical design and `anbv_wajo.txt` for the exhaustive
interviewer walkthrough.
