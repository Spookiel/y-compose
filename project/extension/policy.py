from extension.abstract_policy import GoalOrientedPolicy
from extension.goal import TerminalRegion
from collections import defaultdict
from typing import List, Dict, Tuple, Callable
from extension.proposition import Proposition
from extension.base import RMAX, RMIN, State, Action
from extension.env import StepResult
from extension.algebra import DiscreteQFunction
import random
class WVFMultiGoalAgent(GoalOrientedPolicy):
    def __init__(
        self, 
        terminal_regions: List[TerminalRegion], 
        tasks: Dict[Proposition, Callable[[State], bool]],
        alpha: float = 1,
        gamma: float = 1, # WVF framework often assumes undiscounted (gamma=1), but 0.99 aids convergence
        epsilon: float = 1
    ):

        self.terminal_regions = terminal_regions
        self.tasks = tasks

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # We track all specific goal propositions PLUS the universal bounds
        self.all_propositions: list[Proposition] = list(tasks.keys()) + [Proposition.WVF_MAX, Proposition.WVF_MIN]
        
        # Q-table structure: q_tables[Proposition][(State, Action)] -> float
        self.q_tables: Dict[Proposition, Dict[Tuple[State, Action], float]] = {
            prop: defaultdict(float) for prop in self.all_propositions
        }

    def get_q_function(self, prop: Proposition) -> DiscreteQFunction:
        """Returns a modular QFunction wrapper for a learned proposition's Q-table."""
        return DiscreteQFunction(self.q_tables[prop])

    def get_action(self, state: State, goal: Proposition) -> Action:
        """Returns the epsilon-greedy action for a specific goal."""
        if random.random() < self.epsilon:
            return random.choice(seq=list(Action))
        return self._argmax_q(state, goal)

    def get_exploration_action(self, state: State) -> Action:
        """
        Returns an exploration action based on the maximum Q-value across all goals.
        a = arg max_b (max_t in G Q(s, t, b)) with prob 1-epsilon, else random.
        """
        if random.random() < self.epsilon:
            return random.choice(seq=list(Action))
            
        # 1. Compute max_t Q(s, t, b) for each action b
        action_utilities = {}
        for action in Action:
            # We only consider defined goals for exploration, matching the pseudocode's G
            utilities = [
                self.q_tables[prop][(state, action)] 
                for prop in self.tasks.keys()
            ]
            action_utilities[action] = max(utilities) if utilities else 0.0
            
        # 2. Return argmax over actions
        max_utility = max(action_utilities.values())
        best_actions = [a for a, u in action_utilities.items() if u == max_utility]
        return random.choice(best_actions)

    def _argmax_q(self, state: State, goal: Proposition) -> Action:
        """Helper to find the best action for a given state and goal."""
        q_values = {
            a: self.q_tables[goal][(state, a)] 
            for a in Action
        }
        max_q = max(q_values.values())
        # Break ties randomly to prevent deterministic getting stuck early in training
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def train_off_policy(self, state: State, action: Action, result: StepResult) -> None:
        """
        The core of Algorithm 1: Updates all Q-functions simultaneously from a single step.
        """
        for prop in self.all_propositions:
            # 1. Determine the task-specific algebraic reward (r_hat)
            r_hat = self._compute_wvf_reward(result, prop)
            
            # 2. Calculate the maximum Q-value for the next state
            if result.is_terminal:
                max_next_q = 0.0
            else:
                max_next_q = max(self.q_tables[prop][(result.next_state, a)] for a in Action)
                
            # 3. Apply the TD update
            current_q = self.q_tables[prop][(state, action)]
            td_target = r_hat + (self.gamma * max_next_q)
            td_error = td_target - current_q
            
            self.q_tables[prop][(state, action)] += self.alpha * td_error

    def _compute_wvf_reward(self, result: StepResult, prop: Proposition) -> float:
        """
        Enforces the strictly sparse Boolean reward bounds required by the WVF algebra.
        """
        # If the environment hasn't reached a terminal state, everyone gets the base step penalty.
        if not result.is_terminal:
            return result.base_reward

        # --- TERMINAL STATE LOGIC ---
        
        # WVF_MAX: An omnipotent task where ANY terminal state is considered a success.
        if prop == Proposition.WVF_MAX:
            return RMAX
            
        # WVF_MIN: A pessimistic task where ANY terminal state is considered a failure.
        if prop == Proposition.WVF_MIN:
            return RMIN
            
        # Specific Goal Logic: Success if the terminal state is inside the region, failure otherwise.
        success_predicate = self.tasks[prop]
        if success_predicate(result.next_state):
            return RMAX
        else:
            return RMIN
  