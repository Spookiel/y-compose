from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import random

from extension.base import State, Action
from extension.algebra import QFunction
from extension.dfa_tracker import DFATracker


class EdgePolicy(ABC):
    """
    Unified interface for policies executing an edge transition.
    """
    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass


class QFunctionPolicy(EdgePolicy):
    """
    Wraps a tabular or composed QFunction into an executable policy
    by taking the argmax over available actions.
    """
    def __init__(self, q_function: QFunction):
        self.q_function = q_function

    def get_action(self, state: State) -> Action:
        q_values = {a: self.q_function(state, a) for a in Action}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)


class TaskExecutor:
    """
    Executes a task by orchestrating a DFATracker and a map of edge policies.
    """
    def __init__(
        self, 
        tracker: DFATracker, 
        edge_policies: Dict[Tuple[str, str], EdgePolicy],
        fallback_policy: Optional[EdgePolicy] = None
    ):
        self.tracker = tracker
        self.edge_policies = edge_policies
        self.fallback_policy = fallback_policy

    def get_action(self, state: State) -> Action:
        active_edges = self.tracker.get_active_edge()
        
        if active_edges:
            # If multiple valid edges exist, we could have a combined policy, 
            # but for abstraction we just pick the first valid edge policy.
            # In a fully refined system, LTLfTask might provide a single combined
            # EdgePolicy for the 'state' rather than per-edge, but this preserves the edge mapping.
            for edge in active_edges:
                if edge in self.edge_policies:
                    return self.edge_policies[edge].get_action(state)
            
        if self.fallback_policy:
            return self.fallback_policy.get_action(state)
            
        # Default fallback to random action if stuck
        return random.choice(list(Action))
