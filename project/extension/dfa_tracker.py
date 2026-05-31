import networkx as nx
from typing import Dict, List, Set, Tuple, Any
from extension.base import State
from extension.policy import WVFMultiGoalAgent
import sympy

class DFATracker:
    """
    Tracks the agent's progress through a Deterministic Finite Automaton (DFA) 
    represented as a NetworkX MultiDiGraph.
    """
    def __init__(self, graph: nx.MultiDiGraph, agent: WVFMultiGoalAgent, initial_state: str = None):
        self.graph = graph
        self.agent = agent
        
        # 1. Discover initial state
        if initial_state is not None:
            self.current_state = initial_state
        else:
            # Look for 'init' node or transition from 'init'
            if 'init' in self.graph:
                # Find the successor of 'init'
                successors = list(self.graph.successors('init'))
                if successors:
                    self.current_state = successors[0]
                else:
                    self.current_state = list(self.graph.nodes())[0]
            else:
                # Fallback to node "0" or "1" if they exist
                for candidate in ["0", "1"]:
                    if candidate in self.graph:
                        self.current_state = candidate
                        break
                else:
                    self.current_state = list(self.graph.nodes())[0]

        # 2. Discover accepting states
        # Often marked with doublecircle, but if not, look for sink states with 'true' self-loop
        # that are not 'init'.
        self.accepting_states = {
            node for node, attr in self.graph.nodes(data=True) 
            if attr.get('shape') == 'doublecircle'
        }
        
        if not self.accepting_states:
            # Heuristic: Sink state with 'true' label on self-loop
            for node in self.graph.nodes():
                if node == 'init': continue
                for u, v, attr in self.graph.out_edges(node, data=True):
                    if u == v and attr.get('label', '').strip('"').lower() == 'true':
                        self.accepting_states.add(node)

    def get_active_policy(self) -> QFunction:
        """
        Returns the QFunction for the highest-priority outgoing edge.
        Favors edges that transition to a DIFFERENT state (goal-seeking).
        """
        edges = self.graph.out_edges(self.current_state, data=True)
        
        # 1. Look for advancement edges
        for u, v, attr in edges:
            if v != u:
                guard = attr.get('label', 'True')
                return parse_guard_to_qfunction(guard, self.agent)
        
        # 2. Fallback to self-loops or random
        if edges:
            # Sort to be deterministic if multiple
            edge_list = list(edges)
            guard = edge_list[0][2].get('label', 'True')
            return parse_guard_to_qfunction(guard, self.agent)
        
        from extension.proposition import Proposition
        return self.agent.get_q_function(Proposition.WVF_MAX)

    def step_dfa(self, grid_state: State) -> str:
        """
        Updates the internal DFA state based on the current environment propositions.
        Returns the new DFA state.
        """
        # 1. Evaluate current propositions
        prop_values = {}
        for prop_id, goal_region in self.agent.goals.items():
            prop_values[prop_id.name.lower()] = goal_region.contains(grid_state)
            
        # 2. Check all outgoing transitions from current state
        edges = self.graph.out_edges(self.current_state, data=True)
        for u, v, attr in edges:
            guard_str = attr.get('label', 'True')
            if self._evaluate_guard(guard_str, prop_values):
                self.current_state = v
                break
                
        return self.current_state

    def _evaluate_guard(self, guard_str: str, prop_values: Dict[str, bool]) -> bool:
        """Evaluates a boolean guard string against the current proposition values."""
        if not guard_str:
            return True
            
        clean_guard = guard_str.strip('"').lower()
        if clean_guard == 'true' or clean_guard == '1':
            return True
        if clean_guard == 'false' or clean_guard == '0':
            return False
            
        # Use sympy for robust evaluation
        expr_str = clean_guard.replace('!', '~').replace('&&', '&').replace('||', '|')
        expr = sympy.sympify(expr_str)
        # Substitute values
        subs = {sympy.Symbol(k): v for k, v in prop_values.items()}
        res = expr.subs(subs)
        return bool(res)

    def is_accepted(self) -> bool:
        return self.current_state in self.accepting_states
