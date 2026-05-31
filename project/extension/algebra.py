from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING
from extension.base import State, Action


if TYPE_CHECKING:
    from extension.proposition import Proposition
    from extension.policy import WVFMultiGoalAgent


import sympy


class QFunction(ABC):
    @abstractmethod
    def __call__(self, state: State, action: Action) -> float:
        pass

    def __or__(self, other: 'QFunction') -> 'QFunction':
        return MaxQFunction(self, other)

    def __and__(self, other: 'QFunction') -> 'QFunction':
        return MinQFunction(self, other)

class DiscreteQFunction(QFunction):
    def __init__(self, q_table: Dict[Tuple[State, Action], float]):
        self._q_table = q_table
        
    def __call__(self, state: State, action: Action) -> float:
        return self._q_table.get((state, action), 0.0)

class MaxQFunction(QFunction):
    def __init__(self, q1: QFunction, q2: QFunction):
        self.q1 = q1
        self.q2 = q2
        
    def __call__(self, state: State, action: Action) -> float:
        return max(self.q1(state, action), self.q2(state, action))

class MinQFunction(QFunction):
    def __init__(self, q1: QFunction, q2: QFunction):
        self.q1 = q1
        self.q2 = q2
        
    def __call__(self, state: State, action: Action) -> float:
        return min(self.q1(state, action), self.q2(state, action))

class NegatedQFunction(QFunction):
    def __init__(self, q: QFunction, q_max: QFunction, q_min: QFunction):
        self.q = q
        self.q_max = q_max
        self.q_min = q_min
        
    def __call__(self, state: State, action: Action) -> float:
        return self.q_max(state, action) + self.q_min(state, action) - self.q(state, action)

def xor_q_functions(q1: QFunction, q2: QFunction, q_max: QFunction, q_min: QFunction) -> QFunction:
    not_q1 = NegatedQFunction(q1, q_max, q_min)
    not_q2 = NegatedQFunction(q2, q_max, q_min)
    return (q1 & not_q2) | (not_q1 & q2)


def extract_formal_skill(self_loop_str: str, target_edge_str: str) -> Dict[str, sympy.Expr]:
    """
    Formally extracts Maintain and Trigger conditions from arbitrary Boolean strings.
    """
    # 1. Clean syntax for SymPy (~ is Not, & is And, | is Or). Strip DOT quotes.
    s_clean = self_loop_str.strip('"').replace('!', '~').replace('&&', '&').replace('||', '|')
    t_clean = target_edge_str.strip('"').replace('!', '~').replace('&&', '&').replace('||', '|')

    # 2. Parse into formal SymPy expressions
    try:
        S_expr = sympy.parse_expr(s_clean)
        T_expr = sympy.parse_expr(t_clean)
    except Exception as e:
        raise ValueError(f"SymPy failed to parse the logical string: {e}")

    # 3. Convert the self-loop to Conjunctive Normal Form (AND of ORs)
    S_cnf = sympy.to_cnf(S_expr)

    # 4. Extract individual conjuncts (clauses)
    # If it's an 'And', .args returns the clauses. Otherwise, the whole expression is one clause.
    if isinstance(S_cnf, sympy.And):
        clauses = S_cnf.args
    else:
        clauses = [S_cnf]

    maintain_clauses = []

    # 5. Test each clause against the target edge
    for clause in clauses:
        # If the clause and the target edge do NOT contradict, it is an invariant.
        # satisfiable() returns a dictionary of truth values if true, or False if unsatisfiable.
        if sympy.satisfiable(clause & T_expr) is not False:
            maintain_clauses.append(clause)

    # 6. Compose the final skill
    if maintain_clauses:
        # Recombine the valid maintain clauses
        maintain_expr = sympy.And(*maintain_clauses)
        composed_skill = sympy.And(maintain_expr, T_expr)
    else:
        maintain_expr = None
        composed_skill = T_expr

    return {
        "maintain_logic": maintain_expr,
        "trigger_logic": T_expr,
        "composed_logic": composed_skill
    }


def expr_to_qfunction(expr: sympy.Expr, agent: 'WVFMultiGoalAgent') -> QFunction:
    """
    Parses a boolean string (e.g. 'reach_zone_a & !reach_zone_b') into a composite QFunction.
    Uses sympy for robust parsing and maps symbols to the agent's learned propositions.
    Handles DOT-style labels which might have extra quotes.
    """

    # Map Proposition names (lowercase) to enum values
    from extension.proposition import Proposition
    prop_map = {p.name.lower(): p for p in Proposition}

    def evaluate_expr(e: sympy.Expr) -> QFunction:
        if isinstance(e, sympy.Symbol):
            # Atomic Proposition
            prop_name = str(e).lower()
            if prop_name not in prop_map:
                raise ValueError(f"Unknown proposition symbol in guard: {prop_name}")
            return agent.get_q_function(prop_map[prop_name])
        
        elif isinstance(e, sympy.Not):
            nested = e.args[0]
            if not isinstance(nested, sympy.Symbol):
                raise ValueError(f"Negation of non-propositional atoms is not allowed {nested}")
            # Nested function must be a Proposition by def
            prop_name =  str(nested).lower()
            if prop_name not in prop_map:
                raise ValueError(f"Unknown proposition {prop_name}")
            negated_prop = Proposition.logical_negation(prop_map[prop_name])
            return agent.get_q_function(negated_prop)
        
        elif isinstance(e, sympy.And):
            # A & B & C ...
            children = [evaluate_expr(arg) for arg in e.args]
            res = children[0]
            for other in children[1:]:
                res = res & other
            return res
        
        elif isinstance(e, sympy.Or):
            # A | B | C ...
            children = [evaluate_expr(arg) for arg in e.args]
            res = children[0]
            for other in children[1:]:
                res = res | other
            return res
        
        else:
            raise NotImplementedError(f"Unsupported sympy operation in guard: {type(e)}")

    return evaluate_expr(expr)


if __name__=="__main__":
    print("--- Example 1: Standard ---")
    res1 = extract_formal_skill("avoid_zone_b & ~reach_zone_a", "reach_zone_a")
    print(f"Maintain: {res1['maintain_logic']}")
    print(f"Trigger:  {res1['trigger_logic']}")
    print(f"Composed: {res1['composed_logic']}")

    # --- Example 2: Arbitrarily Complex Structure ---
    # Self-loop requires avoiding B OR C, while waiting for A to happen.
    print("\n--- Example 2: Complex Structure ---")
    res2 = extract_formal_skill("(avoid_b & avoid_c) & ~reach_a", "reach_a")
    print(f"Maintain: {res2['maintain_logic']}")
    print(f"Trigger:  {res2['trigger_logic']}")
    print(f"Composed: {res2['composed_logic']}")