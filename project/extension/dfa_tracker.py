import sympy
from typing import Dict, Optional, List, Tuple, Callable, Any
from extension.base import State
import networkx as nx
from extension.proposition import Proposition

class DFATracker:
    """
    Tracks the agent's progress through a DFA task.
    """
    def __init__(
        self, 
        graph: nx.MultiDiGraph, 
        tasks: Dict[Proposition, Callable[[State], bool]],
        initial_state: Optional[str] = None
    ):
        self.graph = graph
        self.tasks = tasks
        
        self.accepting_states = {
            node for node, attr in self.graph.nodes(data=True) 
            if attr.get('shape') == 'doublecircle'
        }
        if not self.accepting_states:
            # Fallback if doublecircle isn't explicitly set but there is an accepting state
            # This is usually parsed from MONA, assuming node 4 or something, but let's just 
            # allow it to be empty or handle it gracefully.
            pass

        self.current_state = initial_state or self._find_initial_state()
        self.valid_states = self._find_valid_states()

    def _find_valid_states(self) -> set:
        """Finds all states that have a path to an accepting state."""
        valid = set()
        for node in self.graph.nodes():
            for acc in self.accepting_states:
                if node == acc or nx.has_path(self.graph, node, acc):
                    valid.add(node)
                    break
        return valid

    def _find_initial_state(self) -> str:
        """Discovers the initial state from the DFA graph."""
        if 'init' in self.graph:
            successors = list(self.graph.successors('init'))
            if successors:
                return successors[0]
        
        # Fallback: look for node '1' or the first node
        if '1' in self.graph:
            return '1'
        return list(self.graph.nodes())[0]

    def get_active_edge(self) -> Optional[Tuple[str, str]]:
        """
        Returns the (u, v) tuple for the highest-priority outgoing advancement edge 
        from the current state that leads to a valid state.
        Filters out self-loops and paths that lead to dead-ends.
        """
        edges = self.graph.out_edges(self.current_state, data=True)
        
        valid_edges = []
        for u, v, attr in edges:
            if v != u and v in self.valid_states:
                valid_edges.append((u, v))
        
        if valid_edges:
            # For now, just return the first valid advancement edge.
            # TaskExecutor could combine multiple policies if we return a list, 
            # but returning one edge or combining them can be handled at the executor level.
            # Let's return a list of valid edges so the executor can OR them.
            return valid_edges
        
        return None

    def step_dfa(self, grid_state: State) -> str:
        """
        Updates the internal DFA state based on the current grid state.
        Returns the new DFA state.
        """
        # Evaluate propositions
        prop_values = {}
        for prop, predicate in self.tasks.items():
            prop_values[prop.name.lower()] = predicate(grid_state)
            
        # Check all outgoing transitions
        edges = self.graph.out_edges(self.current_state, data=True)
        
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

