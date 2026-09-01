"""Fast behavioral checks for cautious contextual preference learning."""

from __future__ import annotations

from math import isclose

from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    AutonomyTier,
    EmailEnvelope,
    FeedbackType,
    Intent,
    MessagePayload,
    ReplyPayload,
    RiskAssessment,
    SenderBucket,
)
from wajo_agent.learning import (
    ContextualPreferenceLearner,
    InMemoryPreferenceRepository,
    beta_tail_probability,
    build_preference_context,
)
from wajo_agent.policy import PolicyEngine, get_capability


def _archive(email: EmailEnvelope) -> ActionProposal:
    return ActionProposal(
        email_id=email.email_id,
        action_type=ActionType.ARCHIVE,
        intent=Intent.NEWSLETTER,
        summary="Archive the newsletter",
        payload=MessagePayload(kind=ActionType.ARCHIVE, message_id=email.provider_message_id),
    )


def _reply(email: EmailEnvelope) -> ActionProposal:
    return ActionProposal(
        email_id=email.email_id,
        action_type=ActionType.SEND_REPLY,
        intent=Intent.REQUEST,
        summary="Reply to the request",
        payload=ReplyPayload(
            kind=ActionType.SEND_REPLY,
            recipients=("person@outside.com",),
            body="Thanks for the message.",
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checks = 0
    repository = InMemoryPreferenceRepository()
    learner = ContextualPreferenceLearner(repository)
    email = EmailEnvelope(
        provider_message_id="newsletter-1",
        sender="news@example.com",
        subject="Weekly newsletter",
        sender_bucket=SenderBucket.KNOWN_BULK,
    )
    archive = _archive(email)
    archive_context = build_preference_context(email, archive)
    archive_capability = get_capability(ActionType.ARCHIVE)

    cold = learner.recommend(archive_context, archive_capability)
    _require(cold.tier == AutonomyTier.ASK, "cold context did not ask")
    _require((cold.alpha, cold.beta, cold.observations) == (1, 1, 0), "wrong prior")
    checks += 2

    for _ in range(6):
        learner.record(archive_context, FeedbackType.APPROVED)
    notify = learner.recommend(archive_context, archive_capability)
    _require(notify.tier == AutonomyTier.NOTIFY, "six approvals did not reach notify")
    checks += 1

    for _ in range(9):
        learner.record(archive_context, FeedbackType.APPROVED)
    silent = learner.recommend(archive_context, archive_capability)
    _require(silent.tier == AutonomyTier.SILENT, "fifteen approvals did not reach silent")
    _require((silent.alpha, silent.beta, silent.observations) == (16, 1, 15), "wrong evidence")
    checks += 2

    rejected = learner.record(archive_context, FeedbackType.REJECTED)
    after_rejection = learner.recommend(archive_context, archive_capability)
    _require((rejected.beta, rejected.cooldown_remaining) == (4, 5), "wrong rejection weight")
    _require(after_rejection.tier == AutonomyTier.ASK, "rejection did not immediately demote")
    checks += 2

    other_email = email.model_copy(
        update={"email_id": "email-other", "sender": "other@example.com"}
    )
    other_context = build_preference_context(other_email, _archive(other_email))
    isolated = learner.recommend(other_context, archive_capability)
    _require(isolated.tier == AutonomyTier.ASK, "evidence leaked into another context")
    _require(isolated.observations == 0, "another context inherited observations")
    checks += 2

    reply_email = email.model_copy(
        update={"email_id": "email-reply", "sender": "person@outside.com"}
    )
    reply = _reply(reply_email)
    reply_context = build_preference_context(reply_email, reply)
    for _ in range(15):
        learner.record(reply_context, FeedbackType.APPROVED)
    reply_recommendation = learner.recommend(
        reply_context,
        get_capability(ActionType.SEND_REPLY),
    )
    _require(
        reply_recommendation.tier == AutonomyTier.NOTIFY,
        "external capability received learned silent handling",
    )
    final_reply_decision = PolicyEngine().decide(
        reply,
        risk=RiskAssessment(),
        preference_tier=reply_recommendation.tier,
    )
    _require(final_reply_decision.tier == AutonomyTier.ASK, "policy did not retain external floor")
    checks += 2

    _require(isclose(beta_tail_probability(1, 1, 0.70), 0.30), "wrong uniform tail")
    _require(
        isclose(beta_tail_probability(16, 1, 0.90), 1.0 - 0.9**16),
        "wrong learned tail probability",
    )
    checks += 2

    print(f"Learning checks passed: {checks}")


if __name__ == "__main__":
    main()
