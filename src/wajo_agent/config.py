from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: Path
    user_id: str
    planner_model: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=Path(os.getenv("WAJO_DB_PATH", "wajo.db")),
            user_id=os.getenv("WAJO_USER_ID", "local-demo-user"),
            planner_model=os.getenv("WAJO_PLANNER_MODEL", "gpt-5.6-terra"),
        )
