from extension.base import State, Action, StepResult
from abc import ABC, abstractmethod
import torch
import os

class Env(ABC):
    @abstractmethod
    def step(self, state:State, action:Action) -> StepResult:
        ...

class GridWorldEnv(Env):
    STEP_REWARD = -0.1
    def __init__(self, x_min: int, x_max: int, y_min: int, y_max: int):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def step(self, state: State, action: Action) -> StepResult:
        new_state:State = self._apply_physics(state, action)
        outside = not (0 <= new_state.x < self.x_max and 0 <= new_state.y < self.y_max)
        return StepResult(next_state=new_state, base_reward=GridWorldEnv.STEP_REWARD, is_terminal=outside)

    def _apply_physics(self, state: State, action: Action) -> State:
        new_x, new_y = state.x, state.y
        
        if action == Action.UP:
            new_y += 1
        elif action == Action.DOWN:
            new_y -= 1
        elif action == Action.LEFT:
            new_x -= 1
        elif action == Action.RIGHT:
            new_x += 1
        
        return State(x=new_x, y=new_y)

    def save(self, filepath: str) -> None:
        """Saves the environment configuration to a .pt file."""
        config = {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(config, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'GridWorldEnv':
        """Loads and returns a GridWorldEnv from a .pt file."""
        config = torch.load(filepath, weights_only=True)
        return cls(**config)

class TraceableEnv(Env):
    def __init__(self, inner_env: Env):
        super().__init__()
        self.inner_env = inner_env
        self.history = []

    def step(self, state:State, action:Action):
        self.history.append(state)
        return self.inner_env.step(state, action)
        