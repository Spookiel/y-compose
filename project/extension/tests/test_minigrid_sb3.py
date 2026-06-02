import os
import argparse
import time

# Force CPU and disable problematic optimizations
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper, ReseedWrapper,SymbolicObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class FixedLayoutRandomAgentWrapper(gym.Wrapper):
    """
    Wrapper that ensures the map layout and goal position are fixed (via a fixed seed),
    but the agent spawns in a random empty location every time reset is called.
    """
    def __init__(self, env, layout_seed=42):
        super().__init__(env)
        self.layout_seed = layout_seed

    def reset(self, **kwargs):
        # 1. Force the fixed seed to generate the SAME layout and goal position
        kwargs["seed"] = self.layout_seed
        obs, info = self.env.reset(**kwargs)
        
        # 2. Re-randomize the agent's position and direction manually
        # This keeps the grid identical but moves the agent
        self.unwrapped.place_agent() 
        
        # 3. Regenerate the observation since the agent moved
        obs = self.unwrapped.gen_obs()
        return obs, info

def wrap_env(env):
    """
    Chains wrappers to:
    1. Fix layout/goal but randomize agent (FixedLayoutRandomAgentWrapper)
    2. See the whole map (FullyObsWrapper)
    3. Include direction and flatten for MLP (FlatObsWrapper)
    """
    # env = FixedLayoutRandomAgentWrapper(env, layout_seed=42)
    env = SymbolicObsWrapper(env)
    # print(env.reset())
    env = FlatObsWrapper(env) 
    return env

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate SB3 agent on MiniGrid.")
    parser.add_argument("--train", action="store_true", help="Train the model instead of loading it.")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments for training.")
    parser.add_argument("--render", action="store_true", help="Visualize evaluation with 'human' render mode.")
    args = parser.parse_args()

    # Use a descriptive model name for this specific configuration
    model_path = "ppo_minigrid_fourrooms_fixed_goal"
    
    if args.train:
        print(f"Initializing {args.envs} MiniGrid environments in parallel for training...")
        env = make_vec_env(
            "BabyAI-GoToObjS6-v1", 
            n_envs=args.envs, 
            vec_env_cls=SubprocVecEnv,
            wrapper_class=wrap_env,
            # env_kwargs={"max_steps": 80} # Give the agent more time to find the goal
        )

        print("Initializing PPO agent (CPU)...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device="cpu"
        )

        print("Starting training (1,000,000 steps)...")
        model.learn(total_timesteps=1_000_000)

        model.save(model_path)
        print(f"Model saved to {model_path}")
        env.close()
    
    # Evaluation
    print(f"\nStarting evaluation (render={args.render})...")
    
    eval_render_mode = "human" if args.render else "rgb_array"
    eval_env = make_vec_env(
        "BabyAI-GoToObjS6-v1", 
        n_envs=1, 
        vec_env_cls=DummyVecEnv,
        wrapper_class=wrap_env,
        env_kwargs={"render_mode": eval_render_mode}
    )

    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file {model_path}.zip not found. Run with --train first.")
        eval_env.close()
        return

    model = PPO.load(model_path, env=eval_env, device="cpu")

    obs = eval_env.reset()
    for i in range(1000): # Increased evaluation steps
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        
        if args.render:
            eval_env.render()
            time.sleep(0.02)
        
        if rewards[0] > 0:
            print(f"Step {i}: Success! Goal reached.")
        
        if dones[0]:
            print(f"Episode finished at step {i}. Resetting...")
    
    print("Evaluation complete.")
    eval_env.close()

if __name__ == "__main__":
    main()
