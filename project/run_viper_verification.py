import sys
import os
import torch
import numpy as np

# Ensure VIPER and project modules are accessible
sys.path.append(os.path.join(os.getcwd(), "viper/python"))

from reach_avoid_tabular import load_room, Room
from boolean_task import GoalOrientedQLearning
from viper_adapter import RoomGymWrapper, VIPERTeacher, EpsilonGreedyDTPolicy
from viper.core.rl import train_dagger, get_rollout, test_policy
from viper.core.dt import DTPolicy, save_dt_policy, load_dt_policy
from smt_verifier import verify_subgoal_dt
from atomic_task import animate_trace
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

POLICY_SAVE_PATH = "project/static/policy"

def main():
    room_name = "office"
    # Use 8 actions if the pre-trained policy supports it, or stick to 4 consistently
    n_actions = 4
    retrain = True
    room = load_room("saved_disc", f"{room_name}.pt", n_actions)
    
    # Initialize the room's terrain before creating masks
    room.start()

    # Load Pre-trained Q-Policy
    qmodel = GoalOrientedQLearning(room)
    checkpoint = torch.load(f"project/static/policy/{room_name}.pt", weights_only=True)
    qmodel.Q_subgoal = checkpoint["subgoal"]
    qmodel.Q_joint = checkpoint["joint"] # Ensure joint policy is loaded if needed elsewhere

    # Target goal_1 (subgoal index 0)
    gr_idx = 0
    goal_mask = room.goals["goal_1"]
    obstacle_mask = (room.terrain == 0)
    
    env = RoomGymWrapper(room, goal_mask, obstacle_mask)
    teacher = VIPERTeacher(qmodel, gr_idx)
    # Use EpsilonGreedyDTPolicy to improve DAgger distillation performance
    student = EpsilonGreedyDTPolicy(max_depth=20, epsilon=0.1, n_actions=n_actions) 
    
    dt_filename = f"extracted_dt_{room_name}_goal{gr_idx}.pk"
    full_path = os.path.join(POLICY_SAVE_PATH, dt_filename)

    if os.path.exists(full_path) and not retrain:
        print(f"Loading existing policy from {full_path}...")
        best_dt = load_dt_policy(POLICY_SAVE_PATH, dt_filename)
    else:
        # VIPER DAgger Extraction: Mimic tabular Q-values with a Decision Tree
        print("Starting VIPER extraction...")
        best_dt = train_dagger(env, teacher, student, lambda x: x, 
                              max_iters=50, n_batch_rollouts=50, max_samples=300000, 
                              train_frac=0.8, is_reweight=True, n_test_rollouts=20)
        
        print(f"Saving extracted policy to {full_path}...")
        save_dt_policy(best_dt, POLICY_SAVE_PATH, dt_filename)

    print("\n--- Extracted Decision Tree Policy ---")
    print(export_text(best_dt.tree, feature_names=["row", "col"]))

    # Exhaustive testing from every valid cell
    print("\n--- Running Exhaustive Grid Testing ---")
    test_results = test_policy_exhaustive(env, best_dt)
    # Visualize pass/fail distribution
    room.draw_policy(torch.zeros(room.shape + (room.n_actions,)), mask=(torch.tensor(test_results) == 1), fn="exhaustive_test_map")

    # Visualize the extracted decision tree policy
    print("\n--- Visualizing Decision Tree Policy ---")
    visualize_dt_policy(best_dt, room, goal_mask, f"dt_policy_{room_name}_goal{gr_idx}")

    # SMT Formal Verification
    goal_coords = torch.nonzero(goal_mask).tolist()
    print(goal_coords)
    obs_coords = torch.nonzero(obstacle_mask).tolist()
    # obs_coords = []
    print("Obstacle coordinates:", obs_coords)
    start_pos = (10, 3)  # Use a starting point consistent with your experiments
    
    verify_subgoal_dt(best_dt, room.shape, start_pos, goal_coords, obs_coords, room.n_actions, horizon=50)

    # Generate and save rollout animation
    print("\n--- Generating Rollout Animation ---")
    trace = get_rollout(env, best_dt, False)
    trace_points = np.array([step[0] for step in trace])
    print(trace_points)
    # obstacle_mask is used as the avoid_region for the visualization
    animate_trace(obstacle_mask, goal_mask, trace_points)
    print("Animation saved to project/static/training/trace.gif")

def test_policy_exhaustive(env: RoomGymWrapper, policy: DTPolicy, horizon: int = None):
    """Tests the policy from every non-obstacle starting cell in the grid."""
    room = env.unwrapped
    h, w = room.shape
    results = np.zeros((h, w), dtype=int) # 0: Untestable (Wall), 1: Success, -1: Failed
    if horizon is None:
        horizon = env.max_steps
    
    traversable_cells = torch.nonzero(room.terrain > 0).tolist()
    success_count = 0
    
    for r, c in traversable_cells:
        # Initialize environment at this specific location
        obs = env.reset(start_state=np.array([r, c]))
        done = False
        steps = 0
        reached_goal = False
        
        while not done and steps < horizon:
            # Predict using the Decision Tree (wrapped in a batch)
            action = policy.predict(np.array([obs]))[0]
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            if reward == 100.0:
                reached_goal = True
                done = True
            # Collisions are non-terminal to match environment logic
            # This allows the policy to recover and continue towards the goal
                
            steps += 1
            
        results[r, c] = 1 if reached_goal else -1
        if reached_goal: success_count += 1
            
    total_testable = len(traversable_cells)
    print(f"Exhaustive Test Summary: {success_count}/{total_testable} ({success_count/total_testable:.2%}) success rate.")
    return results

def visualize_dt_policy(dt_policy: DTPolicy, room: Room, goal_mask: torch.Tensor, filename: str):
    """Generates a Q-value-like tensor from a DTPolicy for visualization."""
    q_values_for_viz = torch.zeros(room.shape + (room.n_actions,))
    for r in range(room.shape[0]):
        for c in range(room.shape[1]):
            state = np.array([[r, c]])
            action = dt_policy.predict(state)[0]
            q_values_for_viz[r, c, action] = 1.0  # Mark the chosen action with a high value
    room.draw_policy(q_values_for_viz, mask=goal_mask, fn=filename)

if __name__ == "__main__":
    main()
