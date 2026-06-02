from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Any

@dataclass(frozen=True)
class State:
    x: int
    y: int

@dataclass(frozen=True)
class ContinuousState(State):
    """
    A state representation for environments with continuous observation spaces.
    Overrides x and y with dummy values to satisfy the base class, and stores
    the actual observation in a hashable tuple.
    """
    x: int = 0
    y: int = 0
    obs: Tuple[float, ...] = ()

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # Add generic integer actions for Gym compatibility
    ACT_0 = 0
    ACT_1 = 1
    ACT_2 = 2
    ACT_3 = 3

@dataclass(frozen=True)
class StepResult:
    next_state: State
    base_reward: float
    is_terminal: bool

RMAX: int = 1
RMIN: int = -1