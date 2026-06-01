import os
import networkx as nx
import pydot
from typing import Dict, Any, Optional
from ltlf2dfa.parser.ltlf import LTLfParser

from extension.policy import WVFMultiGoalAgent
from extension.algebra import extract_formal_skill, expr_to_qfunction, QFunction
from extension.render import GridWorldRenderer
from extension.executor import EdgePolicy, QFunctionPolicy

class LTLfTask:
    """
    Encapsulates the logic for parsing an LTLf formula, generating a DFA, 
    and composing Q-functions for the edges of the DFA using the agent's learned skills.
    """
    def __init__(self, formula_str: str, agent: WVFMultiGoalAgent):
        self.formula_str = formula_str
        self.agent = agent
        
        self.nx_graph: nx.MultiDiGraph = self._build_dfa_graph()
        self.edge_policies: Dict[tuple, EdgePolicy] = self._compose_edge_policies()

    def _build_dfa_graph(self) -> nx.MultiDiGraph:
        """Parses the LTLf formula and converts it into a NetworkX MultiDiGraph."""
        parser = LTLfParser()
        formula = parser(self.formula_str)
        
        # Get DOT string from ltlf2dfa
        dot_str = formula.to_dfa()
        
        # Convert DOT to NetworkX MultiDiGraph
        pydot_graph = pydot.graph_from_dot_data(dot_str)[0]
        return nx.drawing.nx_pydot.from_pydot(pydot_graph)

    def _compose_edge_policies(self) -> Dict[tuple, EdgePolicy]:
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
                
                # Store it as an executable QFunctionPolicy
                edge_policies[(u, v)] = QFunctionPolicy(q_func)
                
                # Store the composed expr back in the graph attributes for reference
                attr['composed_expr'] = composed_expr
                
        return edge_policies

    def extract_dt_policies(self, env, **viper_kwargs) -> Dict[tuple, EdgePolicy]:
        """
        Uses VIPER to extract Decision Trees for each composed edge policy.
        Only learns for edges leading to 'valid' nodes (nodes that can reach an accepting state).
        Returns a dictionary of DTPolicyWrappers.
        """
        import sys
        viper_path = os.path.abspath(os.path.join(os.getcwd(), "..", "viper", "python"))
        if viper_path not in sys.path:
            sys.path.append(viper_path)
            
        from extension.viper_integration import WVFEnvGymWrapper, QFunctionTeacher, DTPolicyWrapper
        from viper.core.rl import train_dagger
        from viper.core.dt import DTPolicy
        import numpy as np
        import random
        
        # Determine accepting states (manual override or shape=doublecircle)
        accepting_states = viper_kwargs.pop('accepting_states', None)
        if accepting_states is None:
            accepting_states = {
                node for node, attr in self.nx_graph.nodes(data=True) 
                if attr.get('shape') == 'doublecircle'
            }
        else:
            accepting_states = set(accepting_states)

        # Find valid nodes (those that can reach an accepting state)
        valid_nodes = set()
        for node in self.nx_graph.nodes():
            for acc in accepting_states:
                if node == acc or nx.has_path(self.nx_graph, node, acc):
                    valid_nodes.add(node)
                    break

        # Extract student-specific parameters from kwargs
        max_depth = viper_kwargs.pop('max_depth', 40)
        n_actions = viper_kwargs.pop('n_actions', 4)

        dt_policies = {}
        for edge, policy in self.edge_policies.items():
            u, v = edge
            
            # CRITICAL: Only learn for edges leading to valid nodes
            if v not in valid_nodes:
                print(f"Skipping extraction for edge {edge} (leads to invalid node)")
                continue

            if not isinstance(policy, QFunctionPolicy):
                continue
                
            print(f"\n--- Extracting Decision Tree for edge {edge} ---")
            composed_expr = self.nx_graph.get_edge_data(u, v)[0]['composed_expr']
            gym_env = WVFEnvGymWrapper(env, task_expr=composed_expr, reward_fn=self.agent.reward_fn)
            teacher = QFunctionTeacher(policy.q_function)
            student = DTPolicy(max_depth=max_depth)
            
            default_kwargs = {
                'max_iters': 20, 
                'n_batch_rollouts': 20, 
                'max_samples': 100000, 
                'train_frac': 0.8, 
                'is_reweight': True, 
                'n_test_rollouts': 20
            }
            default_kwargs.update(viper_kwargs)
            
            best_dt = train_dagger(gym_env, teacher, student, lambda x: x, **default_kwargs)
            dt_policies[edge] = DTPolicyWrapper(best_dt)
            
        return dt_policies

    def save_dt_policies(self, dt_policies: Dict[tuple, EdgePolicy], filepath: str) -> None:
        """Saves the distilled DT policies to a pickle file."""
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dt_policies, f)
        print(f"DT Policies saved to {filepath}")

    def load_dt_policies(self, filepath: str) -> Dict[tuple, EdgePolicy]:
        """Loads distilled DT policies from a pickle file."""
        import pickle
        # Ensure viper is in path so pickle can find the classes if needed
        import sys
        viper_path = os.path.abspath(os.path.join(os.getcwd(), "..", "viper", "python"))
        if viper_path not in sys.path:
            sys.path.append(viper_path)

        with open(filepath, 'rb') as f:
            dt_policies = pickle.load(f)
        print(f"DT Policies loaded from {filepath}")
        return dt_policies


    def render_edge_policies(self, renderer: GridWorldRenderer, save_dir: str = "static") -> None:
        """Helper to visualize all composed edge policies."""
        os.makedirs(save_dir, exist_ok=True)
        
        for u, v, attr in self.nx_graph.edges(data=True):
            if (u, v) in self.edge_policies:
                composed_expr = attr.get('composed_expr', 'unknown')
                policy = self.edge_policies[(u, v)]
                
                # Only render QFunction policies directly via values
                if isinstance(policy, QFunctionPolicy):
                    q_func = policy.q_function
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
