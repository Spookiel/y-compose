import os
from PIL import Image
import io

from extension.env import GridWorldEnv, TraceableEnv
from extension.base import State, Action
from extension.goal import TerminalRegion
from extension.proposition import Proposition
from extension.render import GridWorldRenderer

def test_create_trace_gif():
    print("--- Creating Trace GIF ---")
    
    # 1. Setup Env
    inner_env = GridWorldEnv(x_min=0, x_max=5, y_min=0, y_max=5)
    env = TraceableEnv(inner_env)
    
    # 2. Setup Goals & Renderer
    goals = [
        TerminalRegion(Proposition.REACH_ZONE_A, lambda s: 4 <= s.x <= 5 and 4 <= s.y <= 5),
    ]
    renderer = GridWorldRenderer(inner_env, goals)
    
    # 3. Simulate Path: (0,0) -> (5,0) -> (5,5)
    current_state = State(x=0, y=0)
    frames = []
    
    # Define movement sequence
    actions = [Action.RIGHT] * 5 + [Action.UP] * 5
    
    # Capture initial frame
    frames.append(capture_frame(renderer, current_state, env.history))
    
    for action in actions:
        res = env.step(current_state, action)
        current_state = res.next_state
        frames.append(capture_frame(renderer, current_state, env.history))
        
    # 4. Save GIF
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, "trace_movement.gif")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"GIF saved to {output_path}")
    assert os.path.exists(output_path)

def capture_frame(renderer, state, history):
    # Use io.BytesIO to avoid saving many temp files
    buf = io.BytesIO()
    renderer.render(state, history=history, path_length=5, save_path=buf)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img
