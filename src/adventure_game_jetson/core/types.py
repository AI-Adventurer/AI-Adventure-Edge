from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class ActionPrediction:
    action: str = "stand"
    confidence: float = 0.0
    produced_at: float = 0.0


@dataclass(slots=True)
class GameEvent:
    prompt: str = ""
    required_action: str = ""
    time_limit_sec: float = 0.0


@dataclass(slots=True)
class GameSnapshot:
    state: str
    hp: int
    score: int
    current_loop: int
    time_left_sec: float
    active_action: str
    narration: str
    event_prompt: str = ""
    result_success: Optional[bool] = None
    game_over: bool = False

