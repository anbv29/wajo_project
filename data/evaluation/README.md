# Evaluation dataset card

Version 1.0.0 is a frozen, deterministic, entirely synthetic benchmark for the proactive email
agent. It contains no copied mailbox data, credentials, or real personal information.

## Files

| File | Contents | Development | Held out |
|---|---:|---:|---:|
| `semantic.jsonl` | 72 labeled planning/risk cases | 48 | 24 |
| `injection.jsonl` | 40 attacks plus 40 matched benign controls | 60 records | 20 records |
| `personas.jsonl` | 4 ordered learning histories, 24 steps each | 3 personas | 1 persona |
| `failures.jsonl` | 24 operational failure contracts | Not split | Not split |

The semantic cases cover every intent in the domain model. An action label can contain more than
one acceptable answer when genuine ambiguity exists. Risk labels specify signals that must be
present rather than requiring the scanner to emit no additional conservative findings.

Each injection attack is paired with a benign message that shares its vocabulary. This makes it
possible to measure attack recall and false positives separately. Pair members always belong to the
same split.

Persona records must be replayed in sequence. A decision at step `t` may use only feedback from
earlier steps. Each step provides the appropriate feedback for an `ASK` decision and for an action
that ran under `NOTIFY` or `SILENT`. The `most_autonomous_allowed` field is a safety ceiling, while
the milestone fields capture the expected first promotion.

Failure records are evaluation contracts, not executable tests by themselves. Step 24 connects
them to deterministic fault injectors. `provider_call_expected` describes whether the injected
failure occurs before or after the mailbox-adapter boundary, and every scenario forbids automatic
retry.

## Governance

- Use development records while changing prompts, rules, or thresholds.
- Do not inspect per-case held-out results while tuning. Run the held-out split only for a candidate
  release and report it separately.
- Keep raw case and pair IDs in reports so errors remain reproducible.
- Treat uncertain high-risk labels conservatively and review them before changing the benchmark.
- Changing any JSONL row requires a deliberate version change or documented refreeze.

`manifest.json` records the version, freeze date, row counts, and SHA-256 of every JSONL file.
`scripts/check_datasets.py` verifies those hashes, parses every row through strict Pydantic schemas,
checks split and pairing invariants, and replays the current offline planner, scanner, policy, and
learner against the labels. `scripts/build_eval_datasets.py --force` is the intentional regeneration
command; it is never run implicitly by the evaluator.

These fixtures are deliberately small and synthetic. They prove reproducibility and safety
plumbing for a take-home project, but they do not establish production accuracy across languages,
cultures, inbox distributions, sophisticated obfuscation, or adversaries adapting to the detector.
