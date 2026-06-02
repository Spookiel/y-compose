import os

# Force CPU and disable problematic optimizations before importing any ML libraries
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from masa.envs.tabular.bridge_crossing import BridgeCrossing, cost_fn, label_fn
from masa.common.utils import make_env

# Ensure directories exist
static_dir = "static"
video_folder = os.path.join(static_dir, "videos")
os.makedirs(video_folder, exist_ok=True)

def build_masa_env():
    """Factory function for the environment."""
    return make_env(
        "BridgeCrossing",
        "LTL_SAFETY",
        400,
        label_fn=label_fn,
        record_video=False,  # Use SB3's VecVideoRecorder for evaluation instead
        env_kwargs={"render_mode": "rgb_array"},
        cost_fn=cost_fn,
        alpha=0.01,
    )

# 1. Preview the environment
print("Generating environment preview...")
preview_env = BridgeCrossing(render_mode="rgb_array", render_window_size=320)
obs, info = preview_env.reset(seed=0)
frame = preview_env.render()
img = Image.fromarray(frame)
preview_path = os.path.join(static_dir, "bridge_preview.png")
img.save(preview_path)
print(f"Preview saved to {preview_path}")
preview_env.close()

# 2. Train the agent
print("\nStarting training (CPU)...")
env = build_masa_env()
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
# model.learn(total_timesteps=100000)  # Short run for verification
model.save("ppo_bridge")
env.close()

# uv pip uninstall triton && PYTHONFAULTHANDLER=1 uv run --no-sync --directory y-compose/project python -m extension.tests.test_masa_bridge
# 3. Evaluate and record video
print("\nStarting evaluation and recording...")
# We build a fresh environment for evaluation to ensure clean state
eval_env = DummyVecEnv([build_masa_env])
eval_env = VecVideoRecorder(
    eval_env, 
    video_folder,
    record_video_trigger=lambda x: True, 
    video_length=200,
    name_prefix="ppo-bridge-eval"
)

model = PPO.load("ppo_bridge", device="cpu")

obs = eval_env.reset()
for i in range(200):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, rewards, dones, info = eval_env.step(action)

eval_env.close()
print(f"\nEvaluation complete. Video saved in {video_folder}")
