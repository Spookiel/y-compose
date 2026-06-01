import matplotlib.pyplot as plt
from PIL import Image
import os

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from masa.envs.tabular.bridge_crossing import BridgeCrossing

# Ensure static directory exists
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

preview_env = BridgeCrossing(render_mode="rgb_array", render_window_size=320)
obs, info = preview_env.reset(seed=0)
print({"reset_obs": obs, "reset_info": info})

# Render the frame
frame = preview_env.render()
img = Image.fromarray(frame)

# Option 1: Save to file (best for CLI environments)
save_path = os.path.join(static_dir, "bridge_preview.png")
img.save(save_path)
print(f"Preview saved to {save_path}")
from pprint import pprint

from masa.common.utils import make_env
from masa.envs.tabular.bridge_crossing import cost_fn, label_fn


def build_masa_env():
    return make_env(
        "ConveyorBelt",
        "LTL_SAFETY",
        400,
        label_fn=label_fn,
        record_video=True,
        env_kwargs = {"render_mode":"rgb_array"},
        video_folder = "bridge-crossing-video",
        cost_fn=cost_fn,
        alpha=0.01,
    )

env = build_masa_env()

obs, info = env.reset(seed=0)

print("reset observation:", obs)
print('info["labels"]:', info["labels"])
print('info["constraint"]:')
pprint(info["constraint"])

ACTION_NAMES = {0: "left", 1: "right", 2: "down", 3: "up", 4: "stay"}
scripted_actions = [1]*10 + [3]*30
rows = []

for step, action in enumerate(scripted_actions, start=1):
    obs, reward, terminated, truncated, info = env.step(action)
    rows.append(
        {
            "step": step,
            "action": ACTION_NAMES[action],
            "obs": int(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "labels": sorted(info["labels"]),
            "constraint_step": info["constraint"]["step"],
        }
    )
    if terminated or truncated:
        break

pprint(rows)
env.close()