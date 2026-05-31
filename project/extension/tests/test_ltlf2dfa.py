import os
import io
import torch
from PIL import Image

from extension.base import State, Action
from extension.env import GridWorldEnv, TraceableEnv
from extension.goal import TerminalRegion
from extension.proposition import Proposition
from extension.policy import WVFMultiGoalAgent
from extension.learning import train_goal_oriented
from extension.render import GridWorldRenderer
from extension.ltlf_task import LTLfTask
from extension.dfa_tracker import DFATracker

def test_sequential_ltlf_policy():
    print("--- Testing Sequential LTLf Policy (Refactored) ---")
    
    # 1. Setup Environment
    env = GridWorldEnv(x_min=0, x_max=7, y_min=0, y_max=7)
    
    # 2. Define Regions & Task Predicates
    goal_a_pred = lambda s: s.x == 3 and 1 <= s.y <= 7
    goal_b_pred = lambda s: s.x == 7 and s.y == 0
    goal_c_pred = lambda s: 1 <= s.x <= 4 and 0 <= s.y <= 2

    # Physical terminal regions for MDP shared dynamics
    terminals = [
        TerminalRegion(Proposition.REACH_ZONE_A, goal_a_pred),
        TerminalRegion(Proposition.REACH_ZONE_B, goal_b_pred),
        TerminalRegion(Proposition.REACH_ZONE_C, goal_c_pred)
    ]

    # Task mapping (including direct learning of negations)
    tasks = {
        Proposition.REACH_ZONE_A: goal_a_pred, 
        Proposition.REACH_ZONE_B: goal_b_pred,
        Proposition.REACH_ZONE_C: goal_c_pred,
        Proposition.AVOID_ZONE_A: lambda s: not goal_a_pred(s), 
        Proposition.AVOID_ZONE_B: lambda s: not goal_b_pred(s),
        Proposition.AVOID_ZONE_C: lambda s: not goal_c_pred(s),
    }
    
    agent = WVFMultiGoalAgent(terminal_regions=terminals, tasks=tasks, epsilon=0.1)

    # 3. Train or Load
    train = False
    static_dir = "static"
    checkpoint_path = os.path.join(static_dir, "wvf_agent_test.pt")

    if os.path.exists(checkpoint_path) and not train:
        print(f"\nLoading pre-trained agent from {checkpoint_path}...")
        agent.load(checkpoint_path)
    else:
        print("\nNo checkpoint found. Training agent...")
        # Increase episodes to handle the wall/8x8 space
        train_goal_oriented(env, agent, episodes=500000, verbose=False)
        print(f"Saving trained agent to {checkpoint_path}...")
        agent.save(checkpoint_path)

    # 4. LTLf Task Composition
    formula_str = "(avoid_zone_b && avoid_zone_c) U (reach_zone_a && X F reach_zone_b)"
    print(f"\nComposing LTLf Task: {formula_str}")
    ltlf_task = LTLfTask(formula_str, agent)
    ltlf_task.print_dfa_info()

    # 5. Visualizing Composition
    static_dir = "static"
    renderer = GridWorldRenderer(env, terminals)
    ltlf_task.render_edge_policies(renderer, save_dir=static_dir)
    
    # 6. Simulation Loop
    # Have to manually pass accepting states due to bug in pydot conversion
    tracker = DFATracker(ltlf_task, list("4"))
    trace_env = TraceableEnv(env)
    current_state = State(x=0, y=0) 
    frames = []
    
    max_steps = 200
    steps = 0
    print(f"\nStarting simulation from {current_state}...")
    
    while steps < max_steps:
        # Get active policy from DFA state
        policy = tracker.get_active_policy()
        print(policy, current_state, tracker.current_state)
        
        # Choose greedy action
        q_values = {a: policy(current_state, a) for a in Action}
        best_action = max(q_values, key=q_values.get)
        
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
            print(f"DFA Transition: {old_dfa_state} -> {new_dfa_state} at step {steps}")
            
        if tracker.is_accepted():
            print(f"LTLf Formula Accepted at step {steps}!")
            break
            
    # Final frame
    frames.append(capture_frame(renderer, current_state, trace_env.history))
    
    # 7. Save Results
    output_path = os.path.join(static_dir, "sequential_task_refactored.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"Simulation GIF saved to {output_path}")
    
    assert tracker.is_accepted(), "Agent failed to complete the task!"

def capture_frame(renderer, state, history):
    buf = io.BytesIO()
    renderer.render(state, history=history, path_length=10, save_path=buf)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

if __name__ == "__main__":
    test_sequential_ltlf_policy()
