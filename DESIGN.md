# Design

## Product contract

The system chooses one of `SILENT`, `NOTIFY`, `ASK`, or `ESCALATE` for every proposed action.
The central invariant is: **autonomy may increase through evidence; authority never increases
through learning.**

## Agent architecture

The `ProactiveEmailAgent` owns a full lifecycle: it observes an email, normalizes it, asks a
planner for a typed proposal, scans risk, loads contextual preference evidence, applies immutable
policy, routes the decision, observes execution, records events, and learns from explicit feedback.

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

## Scope

The required path uses fixtures, SQLite, a mock mailbox executor, and a CLI. An OpenAI Responses
API planner is included behind configuration. Gmail remains an optional adapter after the offline
safety and evaluation gates pass.
