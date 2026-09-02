# Design

## Product contract

The system chooses one of `SILENT`, `NOTIFY`, `ASK`, or `ESCALATE` for every proposed action.
The central invariant is: **autonomy may increase through evidence; authority never increases
through learning.**

## Agent architecture

The `ProactiveEmailAgent` owns a full lifecycle: it observes an email, claims the provider message
against duplicate delivery, normalizes it, scans risk, asks a planner for a typed proposal, loads
contextual preference evidence, applies immutable policy, routes the decision, observes execution,
records events, and waits for explicit feedback before learning.

The planner is intentionally tool-less. Only the executor has side-effect capability, and it is
reachable only through final policy, approval, payload-integrity, and idempotency validation.
Every planner implements `plan(PlannerRequest) -> PlannerOutput`; the output must pass the request
allowlist before application identity is attached. Every mailbox adapter implements only
`execute(ExecutionCommand) -> ExecutionResult`. Contract guards reject disabled or under-floor
commands and results that do not match the exact command identity.

`OfflinePlanner` is the default reproducible planner for fixtures and demos. Ordered local rules
classify a bounded set of intents and produce conservative typed proposals. Sensitive intents
produce `NO_ACTION`, meetings and requests create drafts rather than sending, and unavailable
actions fall back to `NO_ACTION` or a contract error. It has no network or side-effect access.

`OpenAIPlanner` is the semantic implementation. It uses the Responses API with
`text_format=PlannerOutput`, stable developer instructions, normalized email serialized as an
untrusted JSON data object, no tools, and `store=False`. Only completed, parsed, allowlisted output
is accepted. SDK failures, incomplete responses, missing parsed output, and schema violations are
translated into safe planner errors.

## Safety floors

- External writes are never below `ASK`.
- Financial, irreversible, unknown, injection-affected, credential, account-recovery, and legal
  commitment cases escalate.
- Trash is at least `ASK`; draft creation is at least `NOTIFY`.
- Learning cannot mutate capability metadata.
- Policy is checked at decision time and immediately before execution.
- External timeouts become `UNKNOWN` and are never automatically retried.

## Learning

Preferences use a versioned exact-context key: action, intent, sender bucket, hashed normalized
sender identity, recipient scope, and action variant. This prevents trust from leaking between
senders or materially different forms of an action. Each context has a Beta posterior. Explicit
approval increases alpha; rejection or undo increases beta and triggers cooldown. Silence is not
feedback. Promotion requires both minimum observations and posterior tail-probability thresholds.
Only capabilities marked internal, reversible, enabled, non-destructive, and non-financial can
receive a learned `SILENT` recommendation. The learner returns its evidence and reasons; policy
retains final authority.

Feedback enters through two evidence-backed workflows, not a generic learning endpoint. Approval
feedback must match a durable granted, consumed, rejected, or invalidated approval for the exact
proposal version. Post-action feedback must match a successful autonomous execution and its exact
command; undo evidence additionally requires a reversible capability. A semantic SHA-256 key makes
repeated clicks and retried requests exactly-once even when they carry a new UI feedback ID.

SQLite takes a write lock before loading the latest contextual preference. It then commits the
feedback receipt, previous and updated Beta state, preference projection, and both audit events in
one transaction. A duplicate returns the original receipt without adding another observation.

## Durable state and audit

`SQLiteStore` is the durable implementation of the learner's preference repository and the local
append-only audit store. Ordered migrations are tracked through `PRAGMA user_version`: version 1
adds events and preferences, version 2 adds approvals, version 3 adds executions, and version 4 adds
durable agent-run claims. Version 5 adds exactly-once feedback receipts, and version 6 adds typed
agent-outcome read models for restart-safe CLI inspection. A database created by a newer application
is rejected instead of guessed at. Connections enable foreign keys, WAL journal mode, a five-second
busy timeout, and short explicit write transactions.

The `audit_events` table assigns an independent monotonically increasing sequence within each
stream. Database triggers reject updates and deletes, so append-only behavior still holds if code
bypasses the repository. Payloads use deterministic, standards-compliant JSON and typed
`AuditEvent` validation when read. `preference_states` is deliberately mutable because it is a
current-state projection. A combined operation updates that projection and appends its explaining
event in one transaction; either both commit or both roll back. Execution projections are added with
their workflow rather than being prematurely represented by unused tables.

`agent_outcomes` is a read model rather than a new authority source. Run completion and its strict
typed outcome snapshot commit atomically. The CLI reloads that snapshot to recover the exact email,
proposal, risk, decision, and approval IDs, then rechecks live approval and execution tables before
acting. Derived Pydantic display fields are excluded from stored input and recomputed during reload.

## Approval authority

An approval is an expiring, single-use authorization for one exact proposal—not a boolean attached
to an email. Canonical UTF-8 JSON includes a schema version, proposal identity and version, email
identity, action type, and the complete typed action payload. Its SHA-256 digest is stored in the
approval record and compared in constant time before grant, invalidation, edit, or consumption.

Only a self-consistent `ASK` decision for an enabled, non-escalated capability may create a request.
The legal state transitions are `PENDING -> GRANTED -> CONSUMED` plus terminal rejection, expiry,
and invalidation paths. SQLite compare-and-set updates make consumption one-time even with stale
workers. An edit atomically invalidates the old request, increments the proposal version, and creates
a separately expiring replacement after the revised proposal has received a new `ASK` decision.
Every transition and its projection update commit in the same transaction. Execution performs a
fresh policy check and consumes the approval in the same transaction as its durable execution claim.

## Execution safety and idempotency

`ExecutionService` is the only application service that invokes a `MailboxExecutor`. Immediately
before claiming an effect it revalidates the exact proposal and decision against current risk and
preference inputs. `ESCALATE` cannot execute, `ASK` requires a matching granted approval, and lower
tiers cannot smuggle an approval into their command.

The idempotency key is canonical SHA-256 over a versioned document containing mailbox identity,
provider message identity, proposal version, action type, and the complete action payload. Display
text, command UUIDs, and autonomy tier are excluded because they do not change the intended effect.
SQLite migration 3 adds one unique execution projection per effect key.

The service first commits an `EXECUTING` record and `execution.started` event; for `ASK`, that same
transaction compare-and-sets the approval from `GRANTED` to `CONSUMED`. Only after commit does the
adapter run. A completed provider response becomes `SUCCEEDED`, a certified pre-effect failure
becomes `FAILED_SAFE`, and a timeout, invalid response, unexpected exception, or ambiguous provider
outcome becomes `UNKNOWN`. Neither terminal failures nor unknown outcomes retry automatically.
If the process crashes after the claim, the durable `EXECUTING` row blocks a duplicate and requires
reconciliation. The included scripted mock adapter makes each outcome reproducible offline.

## End-to-end orchestration

The agent records the exact stage order `OBSERVE -> NORMALIZE -> ASSESS_RISK -> INTERPRET ->
DECIDE_AUTONOMY -> ACT_OR_WAIT -> LEARN`. The final stage explicitly records that no learning occurs
without user feedback. A unique `(mailbox_id, provider_message_id)` run claim suppresses duplicate
webhook delivery before normalization, planning, approval creation, or execution.

`SILENT` executes without a success notification, `NOTIFY` executes and produces a user message,
`ASK` creates an expiring pending approval without calling the mailbox, and `ESCALATE` performs no
effect. Failed or unknown autonomous effects always surface a message even when their original tier
was `SILENT`. Planner failure also produces a complete auditable escalation rather than permission
to act. Each run has its own ordered event stream containing observation, normalization, risk,
proposal, decision, and completion evidence; approval and execution details remain in their linked
streams.

## Command-line demo

The Typer CLI exposes database initialization, fixture ingestion, inbox and decision inspection,
approval, rejection, proposal editing, explicit execution feedback, preferences, audit events, and
a five-scenario offline demo. Human output uses Rich tables; placing `--json` before the command
returns deterministic machine-readable JSON.

Approval commands never accept a replacement proposal from display output. They load the exact
typed outcome by durable approval or decision ID. Edits create a new proposal version and approval,
re-run planner-output binding and deterministic policy, invalidate the old approval, and record
explicit edit evidence. The shared planner contract also requires every message-scoped action to
target the provider message that was actually observed.

The demo reaches `NOTIFY` and `SILENT` through real evidence-backed feedback operations rather than
seeding counters. It also demonstrates cold-start `ASK`, prompt-injection `ESCALATE`, and an
ambiguous provider result becoming `UNKNOWN` without retry.

## Optional Gmail adapter

Gmail remains an infrastructure adapter behind the existing `MailboxExecutor` protocol. A
dependency-free HTTPS transport accepts an injected OAuth access-token provider and never logs the
token, request body, or email content. `GmailReader` requests one `format=full` message and maps its
headers, text/HTML MIME parts, attachment metadata, provider message ID, and thread ID into the
ordinary `EmailEnvelope`; it never downloads attachment bytes.

`GmailMailboxExecutor` prepares read-state changes, archive, configured labels, trash, drafts, and
approved allowlisted replies using the documented Gmail v1 REST shapes. Permanent deletion and
unsupported external actions fail before a provider call. The adapter requests only
`gmail.modify`, not the broader full-mailbox scope. It is disabled and dry-run by default; a dry-run
returns `FAILED_SAFE` because no real effect occurred. Mutating timeouts, rate limits, server errors,
malformed success responses, and mismatched provider IDs become `UNKNOWN` and are not retried.

The CLI's `gmail-ingest` command permits a single authenticated read and agent analysis while
leaving mutations dry-run. General CLI approval refuses Gmail-backed outcomes so it cannot consume
authority against the mock adapter. Live effects require a dedicated harness to supply
`GmailAdapterConfig(enabled=True, dry_run=False)`, explicit label IDs, an outbound recipient
allowlist, and a dedicated test account.

## Evaluation datasets

`data/evaluation` is a versioned, frozen, synthetic benchmark rather than a collection of ad hoc
demo messages. It contains 72 semantic cases (48 development, 24 held out), 40 injection attacks
paired with 40 vocabulary-matched benign controls, four chronological learning personas with 24
emails each, and 24 failure contracts. Case IDs, expected planner behavior, required risk signals,
minimum autonomy tiers, feedback, and annotation notes are strict Pydantic data.

The manifest records a SHA-256 and row count for each JSONL file. `scripts/check_datasets.py`
rejects drift, malformed rows, broken split/pair invariants, duplicate identities, missing intent
coverage, and unsafe retry contracts. It also runs every semantic and adversarial message through
the real offline planner, normalizer, scanner, and policy, then replays each persona chronologically
through the real contextual learner. This catches labels that look reasonable on paper but do not
match executable behavior. Dataset construction is deterministic and explicit; the normal checker
never regenerates or mutates the frozen benchmark.

## Scope

The required path uses fixtures, SQLite, a mock mailbox executor, and a CLI. An OpenAI Responses
API planner and a disabled-by-default Gmail adapter are included behind configuration. The offline
safety and evaluation gates remain the required submission path.
