import numpy as np
import random
from typing import Callable, Any, Tuple
import torch

# Ensure we can import VIPER (assuming sys.path is handled in the main script)
try:
    from viper.core.dt import DTPolicy
except ImportError:
    # Dummy class for type hinting if VIPER isn't loaded yet
    class DTPolicy:
        def predict(self, obss): pass

from extension.base import State, Action
from extension.env import GridWorldEnv
from extension.algebra import QFunction
from extension.executor import EdgePolicy
from extension.goal import TerminalRegion
from extension.reward import WVFRewardFunction
from typing import List
import sympy

class WVFEnvGymWrapper:
    """
    Wraps the GridWorldEnv to follow the OpenAI Gym interface used by VIPER.
    Converts extension.base.State to np.array and maps actions.
    """
    def __init__(
        self, 
        env: GridWorldEnv, 
        task_expr: sympy.Expr, 
        reward_fn: WVFRewardFunction, 
        max_steps: int = 80, 
        terminal_regions: List[TerminalRegion] = []
    ):
        self.env = env
        self.task_expr = task_expr
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.curr_step = 0
        self.terminal_regions = terminal_regions
    
    def reset(self, start_state=None) -> np.ndarray:
        self.curr_step = 0
        if start_state is not None:
            state = State(x=int(start_state[0]), y=int(start_state[1]))
        else:
            state = State(
                x=random.randint(self.env.x_min, self.env.x_max),
                y=random.randint(self.env.y_min, self.env.y_max)
            )
        # Store for the step function if needed, though env is stateless.
        self.last_state = state
        return np.array([state.x, state.y])

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.curr_step += 1
        
        # Convert integer to Action enum
        action = Action(int(action_idx))
        
        # Step environment
        res = self.env.step(self.last_state, action)
        self.last_state = res.next_state

        
        obs = np.array([res.next_state.x, res.next_state.y])
        
        # Determine specific WVF reward for this composed task edge
        reward = self.reward_fn(res, self.task_expr)
        
        done = res.is_terminal or self.curr_step >= self.max_steps
        
        return obs, reward, done, {}

class QFunctionTeacher:
    """
    Wraps an algebraic QFunction to act as a teacher for VIPER's DAgger algorithm.
    """
    def __init__(self, q_function: QFunction):
        self.q_function = q_function

    def predict(self, obss: np.ndarray) -> np.ndarray:
        """Returns the best action index for a batch of observations."""
        actions = []
        for obs in obss:
            state = State(x=int(obs[0]), y=int(obs[1]))
            q_vals = [self.q_function(state, a) for a in Action]
            max_q = max(q_vals)
            # Find all actions with max_q and pick one randomly
            best_action_indices = [i for i, q in enumerate(q_vals) if q == max_q]
            actions.append(random.choice(best_action_indices))
        return np.array(actions)

    def predict_q(self, obss: np.ndarray) -> np.ndarray:
        """Returns all Q-values for a batch of observations."""
        q_batch = []
        for obs in obss:
            state = State(x=int(obs[0]), y=int(obs[1]))
            q_vals = [self.q_function(state, a) for a in Action]
            q_batch.append(q_vals)
        return np.array(q_batch)

class DTPolicyWrapper(EdgePolicy):
    """
    Wraps a VIPER DTPolicy into our unified EdgePolicy interface.
    """
    def __init__(self, dt_policy: DTPolicy, epsilon: float = 0.0):
        self.dt_policy = dt_policy
        self.epsilon = epsilon

    def get_action(self, state: State) -> Action:
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(list(Action))
            
        obs = np.array([[state.x, state.y]])
        action_idx = self.dt_policy.predict(obs)[0]
        return Action(int(action_idx))
