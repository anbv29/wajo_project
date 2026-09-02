"""The ordered lifecycle followed by every proactive-agent run.

This module deliberately contains no email API, database, or LLM code. It defines what it means
for this project to behave like an agent: observe, normalize, assess risk, interpret, decide, act,
and learn.
Later components will plug into these stages without being allowed to skip the safety stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class AgentStage(StrEnum):
    """One stage in the proactive email agent's reasoning-and-action loop."""

    OBSERVE = "observe"
    NORMALIZE = "normalize"
    ASSESS_RISK = "assess_risk"
    INTERPRET = "interpret"
    DECIDE_AUTONOMY = "decide_autonomy"
    ACT_OR_WAIT = "act_or_wait"
    LEARN = "learn"


STAGE_ORDER: tuple[AgentStage, ...] = tuple(AgentStage)


@dataclass(frozen=True, slots=True)
class AgentTraceEntry:
    """An auditable note showing what happened at one lifecycle stage."""

    stage: AgentStage
    note: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class AgentLifecycle:
    """Immutable progress record for one email-agent run."""

    run_id: str
    email_id: str
    entries: tuple[AgentTraceEntry, ...]

    @classmethod
    def start(cls, email_id: str) -> AgentLifecycle:
        """Start a run in OBSERVE because an agent must perceive before reasoning."""
        return cls(
            run_id=f"run_{uuid4().hex}",
            email_id=email_id,
            entries=(
                AgentTraceEntry(
                    stage=AgentStage.OBSERVE,
                    note="Incoming email was observed as untrusted input.",
                    recorded_at=datetime.now(UTC),
                ),
            ),
        )

    @property
    def current_stage(self) -> AgentStage:
        return self.entries[-1].stage

    @property
    def is_complete(self) -> bool:
        return self.current_stage == AgentStage.LEARN

    def advance(self, stage: AgentStage, note: str) -> AgentLifecycle:
        """Return a new trace at the next stage; skipping safety stages is rejected."""
        if self.is_complete:
            raise ValueError("a completed agent run cannot advance")

        current_index = STAGE_ORDER.index(self.current_stage)
        expected_stage = STAGE_ORDER[current_index + 1]
        if stage != expected_stage:
            raise ValueError(
                f"agent must advance from {self.current_stage} to {expected_stage}, not {stage}"
            )
        if not note.strip():
            raise ValueError("every lifecycle stage requires an audit note")

        entry = AgentTraceEntry(
            stage=stage,
            note=note.strip(),
            recorded_at=datetime.now(UTC),
        )
        return AgentLifecycle(
            run_id=self.run_id,
            email_id=self.email_id,
            entries=(*self.entries, entry),
        )
