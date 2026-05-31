from typing import Callable
from extension.base import State
from extension.proposition import Proposition

class TerminalRegion:
    def __init__(self, prop_id: Proposition, predicate: Callable[[State], bool]):
        self.id = prop_id
        self._predicate = predicate

    def contains(self, state: State) -> bool:
        """Evaluates if the physical state satisfies this goal region."""
        return self._predicate(state)


# Example Instantiation:
# zone_a = GoalRegion(
#     prop_id=Proposition.REACH_ZONE_A, 
#     predicate=lambda s: 5 <= s.x <= 10 and s.y == 0
# )