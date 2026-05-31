import os
from extension.env import GridWorldEnv
from extension.base import State, Action
from extension.goal import GoalRegion
from extension.proposition import Proposition
from extension.render import GridWorldRenderer

def test_gridworld():
    print("--- Testing GridWorldEnv ---")
    
    # 1. Instantiate with bounds
    env = GridWorldEnv(x_min=0, x_max=5, y_min=0, y_max=5)
    print(f"Created env with bounds: x[{env.x_min}, {env.x_max}], y[{env.y_min}, {env.y_max}]")
    
    # 2. Test Physics
    start_state = State(x=0, y=0)
    
    # Move UP
    res_up = env.step(start_state, Action.UP)
    print(f"Move UP from (0,0) -> ({res_up.next_state.x}, {res_up.next_state.y})")
    assert res_up.next_state == State(x=0, y=1)
    
    # Test Boundary (LEFT)
    res_left = env.step(start_state, Action.LEFT)
    print(f"Move LEFT from (0,0) [Boundary] -> ({res_left.next_state.x}, {res_left.next_state.y})")
    assert res_left.next_state == State(x=0, y=0)
    
    # Test Boundary (DOWN)
    res_down = env.step(start_state, Action.DOWN)
    print(f"Move DOWN from (0,0) [Boundary] -> ({res_down.next_state.x}, {res_down.next_state.y})")
    assert res_down.next_state == State(x=0, y=0)

    # 3. Test Save/Load
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    save_path = os.path.join(static_dir, "test_env.pt")
    
    env.save(save_path)
    print(f"Saved env to {save_path}")
    
    loaded_env = GridWorldEnv.load(save_path)
    print(f"Loaded env bounds: x[{loaded_env.x_min}, {loaded_env.x_max}], y[{loaded_env.y_min}, {loaded_env.y_max}]")
    assert loaded_env.x_min == env.x_min
    assert loaded_env.y_max == env.y_max
    
    # 4. Test Rendering
    print("--- Testing GridWorldRenderer ---")
    goals = [
        GoalRegion(Proposition.REACH_ZONE_A, lambda s: 4 <= s.x <= 5 and 4 <= s.y <= 5),
        GoalRegion(Proposition.REACH_ZONE_B, lambda s: 0 <= s.x <= 1 and 4 <= s.y <= 5)
    ]
    
    renderer = GridWorldRenderer(env, goals)
    agent_pos = State(x=2, y=2)
    render_path = os.path.join(static_dir, "test_render.png")
    
    renderer.render(agent_pos, save_path=render_path)
    print(f"Rendered grid to {render_path}")
    
    assert os.path.exists(render_path)
