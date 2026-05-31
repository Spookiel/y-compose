import sympy
from typing import Dict, Optional, List
from extension.base import State
from extension.ltlf_task import LTLfTask
from extension.algebra import QFunction

class DFATracker:
    """
    Tracks the agent's progress through a DFA task.
    Uses an LTLfTask to manage the graph and edge policies.
    """
    def __init__(self, task: LTLfTask, accepting_states: List[str], initial_state: Optional[str] = None):
        self.task = task
        if not accepting_states:
            raise ValueError("DFA should have at least one accepting state")
        self.current_state = initial_state or self._find_initial_state()
        self.accepting_states = accepting_states
        self.valid_states = self._find_valid_states()
        if not self.valid_states:
            raise ValueError("No valid states!")

    def _find_valid_states(self) -> set:
        """Finds all states that have a path to an accepting state."""
        import networkx as nx
        valid = set()
        graph = self.task.nx_graph
        for node in graph.nodes():
            for acc in self.accepting_states:
                if node == acc or nx.has_path(graph, node, acc):
                    valid.add(node)
                    break
        return valid

    def _find_initial_state(self) -> str:
        """Discovers the initial state from the DFA graph."""
        graph = self.task.nx_graph
        if 'init' in graph:
            successors = list(graph.successors('init'))
            if successors:
                return successors[0]
        
        # Fallback: look for node '1' or the first node
        if '1' in graph:
            return '1'
        return list(graph.nodes())[0]

    def get_active_policy(self) -> QFunction:
        """
        Returns the composed QFunction for outgoing edges that lead to a valid state.
        Filters out self-loops and paths that lead to dead-ends.
        """
        graph = self.task.nx_graph
        edges = graph.out_edges(self.current_state, data=True)
        
        valid_policies = []
        for u, v, attr in edges:
            if v != u and v in self.valid_states:
                if (u, v) in self.task.edge_policies:
                    valid_policies.append(self.task.edge_policies[(u, v)])
        
        if valid_policies:
            # If there are multiple valid paths, we OR them (MaxQFunction)
            # This allows the agent to take whichever valid path is easiest.
            combined_policy = valid_policies[0]
            for p in valid_policies[1:]:
                combined_policy = combined_policy | p
            return combined_policy
        
        # If there are no valid advancement edges, we are in a dead-end or already accepted.
        from extension.proposition import Proposition
        if self.is_accepted():
            return self.task.agent.get_q_function(Proposition.WVF_MAX)
        return self.task.agent.get_q_function(Proposition.WVF_MIN)

    def step_dfa(self, grid_state: State) -> str:
        """
        Updates the internal DFA state based on the current grid state.
        Returns the new DFA state.
        """
        # Evaluate propositions
        prop_values = {}
        for prop, predicate in self.task.agent.tasks.items():
            prop_values[prop.name.lower()] = predicate(grid_state)
            
        # Check all outgoing transitions
        graph = self.task.nx_graph
        edges = graph.out_edges(self.current_state, data=True)
        
        for u, v, attr in edges:
            guard_str = attr.get('label', 'true').strip('"').lower()
            if self._evaluate_guard(guard_str, prop_values):
                self.current_state = v
                break
                
        return self.current_state

    def _evaluate_guard(self, guard_str: str, prop_values: Dict[str, bool]) -> bool:
        """Evaluates a boolean guard string against the current proposition values."""
        if not guard_str or guard_str == 'true':
            return True
        if guard_str == 'false':
            return False
            
        # Use sympy for robust evaluation
        expr_str = guard_str.replace('!', '~').replace('&&', '&').replace('||', '|')
        expr = sympy.sympify(expr_str)
        subs = {sympy.Symbol(k): v for k, v in prop_values.items()}
        return bool(expr.subs(subs))

    def is_accepted(self) -> bool:
        return self.current_state in self.accepting_states
