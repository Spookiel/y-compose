import os
import networkx as nx
import pydot
from typing import Dict, Any, Optional
from ltlf2dfa.parser.ltlf import LTLfParser

from extension.policy import WVFMultiGoalAgent
from extension.algebra import extract_formal_skill, expr_to_qfunction, QFunction
from extension.render import GridWorldRenderer

class LTLfTask:
    """
    Encapsulates the logic for parsing an LTLf formula, generating a DFA, 
    and composing Q-functions for the edges of the DFA using the agent's learned skills.
    """
    def __init__(self, formula_str: str, agent: WVFMultiGoalAgent):
        self.formula_str = formula_str
        self.agent = agent
        
        self.nx_graph: nx.MultiDiGraph = self._build_dfa_graph()
        self.edge_policies: Dict[tuple, QFunction] = self._compose_edge_policies()

    def _build_dfa_graph(self) -> nx.MultiDiGraph:
        """Parses the LTLf formula and converts it into a NetworkX MultiDiGraph."""
        parser = LTLfParser()
        formula = parser(self.formula_str)
        
        # Get DOT string from ltlf2dfa
        dot_str = formula.to_dfa()
        print(dot_str)
        # Convert DOT to NetworkX MultiDiGraph
        pydot_graph = pydot.graph_from_dot_data(dot_str)[0]
        return nx.drawing.nx_pydot.from_pydot(pydot_graph)

    def _compose_edge_policies(self) -> Dict[tuple, QFunction]:
        """
        Iterates through the DFA edges, extracts formal skills (maintain/trigger),
        and creates composed QFunctions for advancing edges.
        """
        edge_policies = {}
        
        # 1. Identify self-loops to extract 'maintain' conditions
        self_loops = {}
        for u, v, attr in self.nx_graph.edges(data=True):
            if u == v:
                if u not in self_loops:
                    self_loops[u] = attr.get('label', 'true')
                    
        # 2. Process advancing edges
        for u, v, attr in self.nx_graph.edges(data=True):
            if u != v and u in self_loops:
                edge_label: str = attr.get('label', 'true')
                
                # Extract logical components
                skill_data = extract_formal_skill(self_loops[u], edge_label)
                composed_expr = skill_data['composed_logic']
                
                # Convert symbolic expression to a concrete QFunction
                q_func = expr_to_qfunction(composed_expr, self.agent)
                
                # Store it. Using (u, v) as key.
                edge_policies[(u, v)] = q_func
                
                # Store the composed expr back in the graph attributes for reference
                attr['composed_expr'] = composed_expr
                
        return edge_policies

    def render_edge_policies(self, renderer: GridWorldRenderer, save_dir: str = "static") -> None:
        """Helper to visualize all composed edge policies."""
        os.makedirs(save_dir, exist_ok=True)
        
        for u, v, attr in self.nx_graph.edges(data=True):
            if (u, v) in self.edge_policies:
                composed_expr = attr.get('composed_expr', 'unknown')
                q_func = self.edge_policies[(u, v)]
                
                save_path = os.path.join(save_dir, f"wvf_{composed_expr}.png")
                print(f"  Rendering edge {u} -> {v}: {composed_expr}")
                
                renderer.render_value_function(
                    q_func, 
                    str(composed_expr), 
                    save_path=save_path, 
                    show_policy=True
                )

    def print_dfa_info(self) -> None:
        """Prints basic information about the parsed DFA."""
        print(f"LTLf Formula: {self.formula_str}")
        print("\nDFA Nodes:")
        for node, attr in self.nx_graph.nodes(data=True):
            print(f"  {node}: {attr}")
            
        print("\nDFA Edges:")
        for u, v, attr in self.nx_graph.edges(data=True):
            label = attr.get('label', '')
            composed = attr.get('composed_expr', '')
            if composed:
                print(f"  {u} -> {v}: {label} (Composed: {composed})")
            else:
                print(f"  {u} -> {v}: {label}")
