import os
import sys
import torch
import pytest
import networkx as nx
import pydot
from PIL import Image
import io
from ltlf2dfa.parser.ltlf import LTLfParser

from extension.base import State, Action
from extension.env import GridWorldEnv, TraceableEnv
from extension.goal import TerminalRegion
from extension.proposition import Proposition
from extension.policy import WVFMultiGoalAgent
from extension.learning import train_goal_oriented
from extension.render import GridWorldRenderer
from extension.algebra import extract_formal_skill, expr_to_qfunction

def test_sequential_ltlf_policy():
    print("--- Testing Sequential LTLf Policy (with ltlf2dfa & networkx) ---")
    
    # 1. Setup Environment & Goals
    env = GridWorldEnv(x_min=0, x_max=7, y_min=0, y_max=7)
    
    # 2. Define Goals
    # Goal A at x=3, acting as a partial wall
    goal_a_pred = lambda s: s.x == 3 and 1 <= s.y <= 7
    goal_a = TerminalRegion(Proposition.REACH_ZONE_A, goal_a_pred)

    # Goal B at bottom-right (7,0)
    goal_b_pred =lambda s: s.x == 7 and s.y == 0
    goal_b = TerminalRegion(Proposition.REACH_ZONE_B, goal_b_pred)

        # Goal B at bottom-right (7,0)
    goal_c_pred =lambda s: 6 <= s.x <= 7 and 1 <= s.y <= 2
    goal_c = TerminalRegion(Proposition.REACH_ZONE_C, goal_c_pred)

    
    agent = WVFMultiGoalAgent(terminal_regions=[goal_a, goal_b, goal_c],tasks={
                    Proposition.REACH_ZONE_A: goal_a_pred, 
                    Proposition.REACH_ZONE_B: goal_b_pred,
                    Proposition.REACH_ZONE_C: goal_c_pred,

                    Proposition.AVOID_ZONE_A: lambda s: not goal_a_pred(s), 
                    Proposition.AVOID_ZONE_B: lambda s: not goal_b_pred(s),
                    Proposition.AVOID_ZONE_C: lambda s: not goal_c_pred(s),
                    },epsilon=0.1)
    
    # 3. Train
    print("\nTraining agent on 8x8 grid with wall...")
    # Increase episodes to handle the wall/8x8 space
    train_goal_oriented(env, agent, episodes=500000, verbose=False)


        # 5. Visualization
    renderer = GridWorldRenderer(env, [goal_a, goal_b, goal_c])
    
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
 
    print("\nVisualizing base goals...")
    for prop in agent.q_tables.keys():
        prop_name = prop.name.lower()
        print(f"  Rendering base goal: {prop_name}")
        renderer.render_value_function(
            agent.get_q_function(prop), 
            prop_name, 
            save_path=os.path.join(static_dir, f"wvf_base_{prop_name}.png"), 
            show_policy=True
        )

    # 3. Generate DFA using ltlf2dfa
    formula_str = " (avoid_zone_b && avoid_zone_c) U (reach_zone_a && X F reach_zone_c)"
    parser = LTLfParser()
    formula = parser(formula_str)
    
    # Get DOT string from ltlf2dfa
    dot_str = formula.to_dfa()
    print(dot_str)
    # Convert DOT to NetworkX MultiDiGraph
    pydot_graph = pydot.graph_from_dot_data(dot_str)[0]
    nx_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
    print(type(nx_graph))
    print("\nDFA Nodes:")
    for source_node, attr in nx_graph.nodes(data=True):
        print(f"  {source_node}: {attr}")
    print("\nDFA Edges:")
    selfs = {}
    for u, v, attr in nx_graph.edges(data=True):
        if u == v:
            if u not in selfs:
                selfs[u] = attr.get('label')
    for u, v, attr in nx_graph.edges(data=True):
        if u != v and u in selfs:
            prop: str = attr.get('label')
            composed = extract_formal_skill(selfs[u], prop)['composed_logic']
            print(f"  {u} -> {v}: {prop} composed {composed}")

            renderer.render_value_function(expr_to_qfunction(composed, agent), f"{composed}", save_path=os.path.join(static_dir, f"wvf_{composed}.png"), show_policy=True)
    
    # The initial state is automatically discovered from 'init' node
    # tracker = DFATracker(nx_graph, agent)
    # print(f"DFA Initial State: {tracker.current_state}")
    
    exit(0)
    # 4. Simulation Loop
    trace_env = TraceableEnv(env)
    current_state = State(x=0, y=0) # Start bottom-right
    frames = []
    renderer = GridWorldRenderer(env, [goal_a, goal_b])
    
    max_steps = 200
    steps = 0
    print(f"\nStarting simulation from {current_state}...")
    
    while steps < max_steps:
        # Get active policy from DFA edge
        edges = tracker.graph.out_edges(tracker.current_state, data=True)
        # Find advancement edge for logging
        adv_edge = next((v, attr.get('label')) for u, v, attr in edges if v != tracker.current_state)
        policy = tracker.get_active_policy()
        
        if steps % 10 == 0:
            print(f"Step {steps}: Current grid state {current_state} DFA State={tracker.current_state}, Target DFA State={adv_edge[0]}, Goal={adv_edge[1]}")
            
        # Choose greedy action
        q_values = {a: policy(current_state, a) for a in Action}
        best_action = max(q_values, key=q_values.get)
        print(best_action)
        # Capture frame
        frames.append(capture_frame(renderer, current_state, trace_env.history))
        
        # Step env
        res = trace_env.step(current_state, best_action)
        current_state = res.next_state
        steps += 1
        
        # Step DFA
        old_dfa_state = tracker.current_state
        new_dfa_state = tracker.step_dfa(current_state)
        if old_dfa_state != new_dfa_state:
            print(f"DFA Transition: {old_dfa_state} -> {new_dfa_state} at step {steps} (Condition Met!)")
            
        if tracker.is_accepted():
            print(f"LTLf Formula Accepted at step {steps}!")
            break
            
    # Final frame
    frames.append(capture_frame(renderer, current_state, trace_env.history))
    
    # 5. Save GIF
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, "sequential_task_new.gif")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"Sequential Task GIF saved to {output_path}")
    
    assert tracker.is_accepted(), "Agent failed to complete the sequential task!"

def capture_frame(renderer, state, history):
    buf = io.BytesIO()
    renderer.render(state, history=history, path_length=10, save_path=buf)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

if __name__ == "__main__":
    test_sequential_ltlf_policy()
