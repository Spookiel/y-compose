import sympy
from typing import Dict, Callable, Union

from extension.base import RMAX, RMIN, StepResult, State
from extension.proposition import Proposition

class WVFRewardFunction:
    """
    Evaluates the WVF reward based on the physical environment outcome
    and the logical task semantics.
    """
    def __init__(self, tasks: Dict[Proposition, Callable[[State], bool]]):
        self.tasks = tasks

    def __call__(self, result: StepResult, task: Union[Proposition, sympy.Expr]) -> float:
        """
        Enforces the strictly sparse Boolean reward bounds required by the WVF algebra.
        """
        # If the environment hasn't reached a terminal state, everyone gets the base step penalty.
        if not result.is_terminal:
            return result.base_reward

        # --- TERMINAL STATE LOGIC ---
        
        if isinstance(task, Proposition):
            # WVF_MAX: An omnipotent task where ANY terminal state is considered a success.
            if task == Proposition.WVF_MAX:
                return RMAX
                
            # WVF_MIN: A pessimistic task where ANY terminal state is considered a failure.
            if task == Proposition.WVF_MIN:
                return RMIN
                
            # Specific Goal Logic: Success if the terminal state is inside the region, failure otherwise.
            if task in self.tasks:
                success_predicate = self.tasks[task]
                if success_predicate(result.next_state):
                    return RMAX
                return RMIN
            raise ValueError(f"Unknown Proposition type {task}")
            
        if hasattr(task, 'subs'):
            # Evaluate composed logic on the terminal state
            subs = {}
            for p, pred in self.tasks.items():
                subs[sympy.Symbol(p.name.lower())] = pred(result.next_state)
            
            is_success = bool(task.subs(subs))
            return RMAX if is_success else RMIN
            
        raise ValueError(f"Unknown task type for reward evaluation: {type(task)}")
