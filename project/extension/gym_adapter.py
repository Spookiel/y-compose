import gymnasium as gym
import numpy as np
from typing import Dict, Callable

from extension.base import ContinuousState, Action, StepResult
from extension.env import Env

class GymAdapter(Env):
    """
    Adapts a standard Gymnasium environment to our custom Env interface.
    Handles continuous states by wrapping observations in ContinuousState.
    """
    def __init__(self, gym_env: gym.Env):
        super().__init__()
        self.gym_env = gym_env
        self.last_obs = None
        
    def reset(self, seed=None):
        obs, info = self.gym_env.reset(seed=seed)
        self.last_obs = obs
        return self._make_state(obs)
        
    def _make_state(self, obs: np.ndarray) -> ContinuousState:
        # Convert numpy array to tuple for hashability in dictionaries/sets
        return ContinuousState(obs=tuple(obs.tolist()))
        
    def step(self, state: ContinuousState, action: Action) -> StepResult:
        """
        Steps the underlying gymnasium environment. 
        Note: Gymnasium environments are stateful. This adapter assumes `step()`
        is called sequentially, and the `state` parameter here is primarily for
        API compatibility, relying on internal gym state.
        """
        # Extract integer action if it's an Action enum
        act_val = action.value if hasattr(action, 'value') else action
        
        obs, reward, terminated, truncated, info = self.gym_env.step(act_val)
        self.last_obs = obs
        
        is_terminal = terminated or truncated
        next_state = self._make_state(obs)
        
        return StepResult(
            next_state=next_state,
            base_reward=float(reward),
            is_terminal=bool(is_terminal)
        )
