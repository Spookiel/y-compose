import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List

from extension.env import GridWorldEnv
from extension.sb3_adapter import SB3EnvAdapter
from extension.goal import TerminalRegion
from extension.proposition import Proposition
from extension.base import State, Action
from extension.render import GridWorldRenderer

from stable_baselines3 import DQN

class SB3QWrapper:
    """Wraps an SB3 DQN model to expose the learned Q-values to the renderer."""
    def __init__(self, model):
        self.model = model
        
    def __call__(self, state: State, action: Action) -> float:
        # Prepare observation batch of size 1
        obs = np.array([[state.x, state.y]], dtype=np.float32)
        obs_tensor = torch.as_tensor(obs).to(self.model.device)
        
        with torch.no_grad():
            # Extract Q-values from the DQN's Q-network
            q_values = self.model.q_net(obs_tensor)
            
        # Return the Q-value for the specific action
        return q_values[0, action.value].item()

def run_evaluation(model, env, renderer, episodes=5, render=False, save_name=None):
    print(f"\nStarting Evaluation of {save_name}...")
    
    # Use existing renderer components for the trajectory plot
    fig, ax = plt.subplots(figsize=(10, 10))
    renderer._setup_axes(ax)
    renderer._draw_base_grid(ax)
    renderer._draw_goals(ax)
    
    successes = 0
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, episodes))
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        path = [State(int(obs[0]), int(obs[1]))]
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            current_state = State(int(obs[0]), int(obs[1]))
            path.append(current_state)
            
        if reward > 0:
            successes += 1
            status = "Success"
        else:
            status = "Failed"
            
        print(f"Episode {ep+1}: {status} in {len(path)-1} steps.")
        
        # Plot trajectory
        xs = [s.x + 0.5 for s in path]
        ys = [s.y + 0.5 for s in path]
        ax.plot(xs, ys, marker='o', markersize=4, label=f"Ep {ep+1} ({status})", color=colors[ep], alpha=0.8)
        # Mark start
        ax.scatter(xs[0], ys[0], marker='s', s=100, color=colors[ep], edgecolors='black', zorder=10)

    ax.set_title(f"DQN Trajectories: {save_name}\nSuccess Rate: {successes}/{episodes}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    
    if save_name:
        plot_path = os.path.join(static_dir, f"{save_name}_paths.png")
        plt.savefig(plot_path)
        print(f"Path visualization saved to {plot_path}")
    
    if render:
        plt.show()
    else:
        plt.close()

    # Use the existing renderer for Policy and Value maps
    print("Generating Policy and Value maps using actual Q-Values from DQN...")
    q_wrapper = SB3QWrapper(model)
    
    renderer.render_policy(
        q_wrapper, 
        title=f"SB3 DQN Policy ({save_name})", 
        save_path=os.path.join(static_dir, f"{save_name}_policy.png")
    )
    renderer.render_value_function(
        q_wrapper, 
        title=f"SB3 DQN Value Function ({save_name})", 
        show_policy=True,
        save_path=os.path.join(static_dir, f"{save_name}_value.png")
    )

    print(f"Evaluation complete. Success Rate: {successes}/{episodes}")

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate SB3 DQN agent on GridWorld.")
    parser.add_argument("--train", action="store_true", help="Train the model from scratch.")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps.")
    parser.add_argument("--render", action="store_true", help="Display the plots after evaluation.")
    args = parser.parse_args()

    # 1. Setup Base Environment (10x10 grid)
    base_env = GridWorldEnv(x_min=0, x_max=10, y_min=0, y_max=10)
    
    # 2. Define Regions
    zone_a_pred = lambda s: 8 <= s.x < 10 and 8 <= s.y < 10
    zone_b_pred = lambda s: 0 <= s.x < 2 and 0 <= s.y < 2
    
    tasks = {
        Proposition.REACH_ZONE_A: zone_a_pred,
        Proposition.REACH_ZONE_B: zone_b_pred
    }
    
    terminals = [
        TerminalRegion(Proposition.REACH_ZONE_A, zone_a_pred),
        TerminalRegion(Proposition.REACH_ZONE_B, zone_b_pred)
    ]
    
    # 3. Target Reach Zone A
    target = Proposition.REACH_ZONE_A
    model_dir = "static/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"dqn_{target.name.lower()}")
    
    # 4. Wrap with Gymnasium Adapter
    env = SB3EnvAdapter(
        base_env=base_env,
        tasks=tasks,
        terminal_regions=terminals,
        target_prop=target,
        max_steps=50
    )
    
    renderer = GridWorldRenderer(base_env, terminals)

    if args.train:
        print(f"Training DQN to reach {target.name} for {args.timesteps} steps...")
        # Configure DQN for tabular-like environment
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=0, 
            device="cpu",
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.2, # Explore for 20% of the training time
            exploration_final_eps=0.05
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        if os.path.exists(model_path + ".zip"):
            print(f"Loading existing model from {model_path}...")
            model = DQN.load(model_path, env=env, device="cpu")
        else:
            print(f"No trained model found at {model_path}. Please run with --train first.")
            return

    # 5. Run Evaluation and Visualize
    run_evaluation(model, env, renderer, episodes=10, render=args.render, save_name=f"dqn_{target.name.lower()}")
    
    env.close()

if __name__ == "__main__":
    main()
