from extension.base import State, Action, StepResult
from abc import ABC, abstractmethod
import torch
import os
from enum import Enum, auto

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
        outside = not (self.x_min <= new_state.x < self.x_max and self.y_min <= new_state.y < self.y_max)
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

class FourRoomsEnv(GridWorldEnv):
    """
    A 4-rooms layout gridworld. Obstacles (walls) are part of the physics dynamics.
    If the agent tries to move into a wall or out of bounds, it stays in place.
    """
    class Proposition(Enum):
        REACH_TOP_LEFT = auto()
        REACH_TOP_RIGHT = auto()
        REACH_BOTTOM_LEFT = auto()
        REACH_BOTTOM_RIGHT = auto()

    def __init__(self, x_max: int = 11, y_max: int = 11):
        # Default 11x11 grid (0..10)
        super().__init__(x_min=0, x_max=x_max, y_min=0, y_max=y_max)
        self._build_walls()

    def _build_walls(self):
        self.walls = set()
        mid_x = self.x_max // 2
        mid_y = self.y_max // 2
        
        # Vertical wall with two doors
        for y in range(self.y_min, self.y_max):
            if y not in [mid_y // 2, mid_y + (mid_y // 2)]:
                self.walls.add((mid_x, y))
                
        # Horizontal wall with two doors
        for x in range(self.x_min, self.x_max):
            if x not in [mid_x // 2, mid_x + (mid_x // 2)]:
                self.walls.add((x, mid_y))

    def step(self, state: State, action: Action) -> StepResult:
        new_state = self._apply_physics(state, action)
        
        # Dynamics-based obstacles: boundaries and internal walls block movement
        if not (self.x_min <= new_state.x < self.x_max and self.y_min <= new_state.y < self.y_max):
            new_state = state # Boundary collision
        elif (new_state.x, new_state.y) in self.walls:
            new_state = state # Wall collision
            
        # In FourRooms, terminality is defined by GoalRegions in the agent/adapter, 
        # not the physics engine itself.
        return StepResult(next_state=new_state, base_reward=self.STEP_REWARD, is_terminal=False)

class TraceableEnv(Env):
    def __init__(self, inner_env: Env):
        super().__init__()
        self.inner_env = inner_env
        self.history = []

    def step(self, state:State, action:Action):
        self.history.append(state)
        return self.inner_env.step(state, action)
        