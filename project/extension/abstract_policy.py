from abc import ABC, abstractmethod
from extension.base import State, Action
from extension.proposition import Proposition

# -- Hierarchy 1: Pure Policies --
class Policy(ABC):
    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass

# -- Hierarchy 2: Goal-Conditioned Policies --
class GoalOrientedPolicy(ABC):
    @abstractmethod
    def get_action(self, state: State, goal: Proposition) -> Action:
        pass

