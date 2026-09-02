# Optional Gmail test-account setup

The take-home works completely offline. Gmail is an optional adapter and should be connected only
to a dedicated test account containing synthetic mail.

## Safety defaults

- Live Gmail effects are disabled by default.
- `GmailAdapterConfig()` is dry-run and never sends a mutating request.
- `wajo gmail-ingest` performs one authenticated read, then analyzes the message with Gmail effects
  remaining in dry-run.
- The CLI refuses to approve a Gmail-backed run; it cannot silently substitute the mock adapter.
- Permanent deletion, payment, account changes, forwarding, and unsubscribe are not implemented.
- Sending requires an `ASK` command with an exact approval ID and a configured recipient allowlist.
- OAuth tokens and client-secret files are ignored by Git.

## Google configuration

1. Create or select a Google Cloud project used only for this test.
2. Enable the Gmail API.
3. Configure the OAuth consent screen for the dedicated test user.
4. Request only `https://www.googleapis.com/auth/gmail.modify` for this adapter. Do not request
   `https://mail.google.com/`, because Wajo never needs immediate permanent deletion.
5. Obtain a short-lived OAuth access token outside this repository using your normal local OAuth
   tooling or secret manager.
6. Inject that token into the process as `WAJO_GMAIL_ACCESS_TOKEN`. Do not paste it into a fixture,
   source file, command argument, screenshot, log, or Git commit.

Then perform a read-only/manual ingestion:

```powershell
uv run wajo --db gmail-test.db gmail-ingest <gmail-message-id>
```

The access-token provider is an interface, so a production integration can supply automatic token
refresh without changing the reader, executor, agent, policy, approval, or learning code.

## Enabling the adapter in controlled code

Construct `GmailMailboxExecutor` with `GmailAdapterConfig(enabled=True, dry_run=False, ...)` only
inside a dedicated live test harness. Configure every allowed outbound recipient and every custom
label name-to-ID mapping explicitly. Continue to pass the executor through `ExecutionService`; do
not call it directly from the planner.

Run the credential-free contract checks first:

```powershell
uv run python scripts/check_gmail.py
```
