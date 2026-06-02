import gymnasium as gym
import numpy as np
from typing import Dict, Callable, List, Optional, Union
import random
from enum import Enum

from extension.base import State, Action, StepResult
from extension.env import Env, GridWorldEnv
from extension.goal import TerminalRegion
from extension.reward import WVFRewardFunction

class SB3EnvAdapter(gym.Env):
    """
    Adapts our custom Env (like GridWorldEnv) to the Gymnasium interface
    so we can train Stable-Baselines3 agents on it.
    
    This adapter is instantiated with a specific target Proposition. 
    To train multiple propositions, we can instantiate multiple environments,
    each focusing on one atomic proposition (or its negation).
    """
    def __init__(
        self, 
        base_env: GridWorldEnv, 
        tasks: Dict[Enum, Callable[[State], bool]],
        terminal_regions: List[TerminalRegion],
        target_prop: Union[Enum, str],
        max_steps: int = 100
    ):
        super().__init__()
        self.base_env = base_env
        self.tasks = tasks
        self.terminal_regions = terminal_regions
        self.target_prop = target_prop
        self.reward_fn = WVFRewardFunction(tasks)
        self.max_steps = max_steps
        
        # SB3 usually prefers float32 for Box spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([self.base_env.x_min, self.base_env.y_min], dtype=np.float32),
            high=np.array([self.base_env.x_max, self.base_env.y_max], dtype=np.float32),
            dtype=np.float32
        )
        
        # We have 4 discrete actions (UP, DOWN, LEFT, RIGHT)
        self.action_space = gym.spaces.Discrete(4)
        
        self.current_state = None
        self.steps = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Standardize seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Random initial state within bounds
        x = random.randint(self.base_env.x_min, self.base_env.x_max - 1)
        y = random.randint(self.base_env.y_min, self.base_env.y_max - 1)
        self.current_state = State(x, y)
        
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1
        
        # Convert integer to our Action enum
        act_enum = Action(action)
        
        # Step the base environment
        result = self.base_env.step(self.current_state, act_enum)
        
        # -- Symbolic Bridge: Determine Terminality --
        # In this framework, the environment is a pure physics engine.
        # Terminality is defined by the agent's goal regions.
        is_goal_terminal = any(region.contains(result.next_state) for region in self.terminal_regions)
        
        if is_goal_terminal:
            result = StepResult(
                next_state=result.next_state,
                base_reward=result.base_reward,
                is_terminal=True
            )
            
        # Compute the specific reward for our target proposition using the WVF reward algebra
        reward = self.reward_fn(result, self.target_prop)
        
        self.current_state = result.next_state
        
        terminated = result.is_terminal
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.current_state.x, self.current_state.y], dtype=np.float32)
