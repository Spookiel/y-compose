import torch
import random
import sympy
import numpy as np
from reach_avoid_tabular import load_room
from boolean_task import GoalOrientedQLearning
from dfa_task import DFA_Task
from product_mdp import ProductMDPGenerator
from wvf_composition import wvf_pointwise_max, wvf_pointwise_min
from atomic_task import animate_trace, AtomicTask
from edge_task import DFA_Edge

def run_minimal_pipeline():
    # 1. Initialize Environment
    elk_name = "office"
    print(f"Loading environment: {elk_name}.pt")
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    start_state = (10, 3) # from original learning.py
    room.start(start_state=start_state)
    
    # 2. Translate LTLf to DFA
    formula = "F(goal_1) & F(goal_2)"
    atomic_formulas = {"goal_1": "F(goal_1)", "goal_2": "F(goal_2)"}
    print(f"Translating LTLf formula to DFA: {formula}")
    atomic_tasks = {name: AtomicTask(af_formula, room, name) for name, af_formula in atomic_formulas.items()}
    dfa_task = DFA_Task(formula, atomic_tasks, name="minimal_test")
    
    # 3. Load Base WVFs
    print("Loading pre-trained Q-learning models...")
    qmodel = GoalOrientedQLearning(room)
    try:
        policy = torch.load(f"project/static/policy/{elk_name}.pt", weights_only=True)
        qmodel.Q_joint = policy["joint"]
        qmodel.Q_subgoal = policy["subgoal"]
        print("Pre-trained models loaded successfully.")
    except FileNotFoundError:
        print("Pre-trained models not found. Running training episodes... (This might take a while)")
        qmodel.train_episodes(num_episodes=85, num_iterations=4, max_steps_per_episode=85)
        torch.save({"joint": qmodel.Q_joint, "subgoal": qmodel.Q_subgoal}, f"project/static/policy/{elk_name}.pt")
        print("Training complete.")

    # 4 & 5. Execution Loop
    print("Starting execution loop...")
    current_state = tuple(start_state)
    current_q = 0 # Initial DFA state
    
    accepting_states = dfa_task.accepting_states
    rejecting_states = dfa_task.rejecting_states
    
    max_steps = 100
    step = 0
    
    print(f"Start State: Env={current_state}, DFA={current_q}")
    while current_q not in accepting_states and step < max_steps:
        # Get valid DFA edges out of current_q
        row = dfa_task.dfa_matrix[current_q]
        valid_transitions = []
        for next_q, edge_formulas in enumerate(row):
            if next_q in rejecting_states:
                continue
            for formula in edge_formulas:
                if formula == sympy.false:
                    continue
                valid_transitions.append((next_q, formula))
        
        if not valid_transitions:
            print("Dead end reached.")
            break
            
        # Randomly select a valid transition
        next_q, selected_formula = random.choice(valid_transitions)
        print(f"Selected Transition to DFA state {next_q} via edge {selected_formula}")
        
        # Execute the task for the selected edge
        edge = DFA_Edge(selected_formula)
        try:
            if selected_formula == sympy.true:
                # If true transition, just step and proceed (or stay in place)
                sequence, trace_segments = [], [[]]
            else:
                sequence, trace_segments = edge.policy_composition(qmodel, atomic_tasks, np.array(current_state))
        except ValueError as e:
            print(f"Error computing policy for edge {selected_formula}: {e}")
            break

        print(f"Executing sequence of subgoals: {sequence}")
        for segment in trace_segments:
            for trace_step in segment:
                if trace_step.action >= 0:
                    next_physical_state, _ = room.step(trace_step.action, trace=True)
                    current_state = tuple(next_physical_state)
                    step += 1

        current_q = next_q
        print(f"Completed edge. New State: Env -> {current_state}, DFA -> {current_q}")

    if current_q in accepting_states:
        print("Pipeline Execution Complete: Accepting State Reached!")
    else:
        print("Pipeline Execution Complete: Max steps reached without acceptance.")

    print("\n--- Generating Rollout Animation ---")
    trace_points = room.get_trace()
    obstacle_mask = (room.terrain == 0)
    # Combine the goals for the visualization (goal_1 and goal_2)
    goal_mask = room.goals["goal_1"] | room.goals["goal_2"]
    
    animate_trace(obstacle_mask, goal_mask, trace_points)
    print("Animation generated successfully. Check project/static/training/trace.gif")

if __name__ == "__main__":
    run_minimal_pipeline()
