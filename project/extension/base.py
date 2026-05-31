from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    x: int
    y: int

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass(frozen=True)
class StepResult:
    next_state: State
    base_reward: float
    is_terminal: bool

RMAX: int = 1
RMIN: int = -100