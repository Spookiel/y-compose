import os
import gymnasium as gym
from PIL import Image

from extension.base import ContinuousState, Action
from extension.gym_adapter import GymAdapter
from extension.goal import TerminalRegion
from extension.proposition import Proposition
from extension.dfa_tracker import DFATracker
from extension.ltlf_task import LTLfTask
from extension.policy import WVFMultiGoalAgent

def test_masa_lunarlander():
    print("--- Testing MASA LunarLander Integration ---")
    
    # 1. Setup Gymnasium Environment with rendering enabled
    raw_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = GymAdapter(raw_env)
    
    # Reset to get initial state
    current_state = env.reset()
    print(f"Initial state type: {type(current_state)}")
    print(f"Observation shape/length: {len(current_state.obs)}")
    
    # 2. Define MASA-style logical labels (Predicates on ContinuousState)
    # LunarLander-v3 observation:
    # 0: x position
    # 1: y position
    # 2: x velocity
    # 3: y velocity
    # 4: angle
    # 5: angular velocity
    # 6: left leg contact (boolean)
    # 7: right leg contact (boolean)
    
    # Proposition A: Successfully landed (both legs touching, upright, slow, between flags)
    # The flags are roughly between x = -0.2 and x = 0.2
    def landed_pred(s: ContinuousState) -> bool:
        obs = s.obs
        x_pos, y_pos = obs[0], obs[1]
        v_y = obs[3]
        angle = obs[4]
        left_leg, right_leg = obs[6], obs[7]
        return (abs(x_pos) < 0.2 and 
                left_leg == 1.0 and 
                right_leg == 1.0 and 
                abs(angle) < 0.1 and 
                abs(v_y) < 0.1)

    # Proposition B: Crashed (very negative y, or tilted too far)
    def crashed_pred(s: ContinuousState) -> bool:
        obs = s.obs
        y_pos = obs[1]
        angle = obs[4]
        return y_pos < -1.0 or abs(angle) > 1.5

    # Note: Gymnasium's step returns terminated=True for landing or crashing anyway,
    # so these regions align with the MDP's natural terminal states.
    terminals = [
        TerminalRegion(Proposition.REACH_ZONE_A, landed_pred),
        TerminalRegion(Proposition.REACH_ZONE_B, crashed_pred)
    ]

    tasks = {
        Proposition.REACH_ZONE_A: landed_pred,
        Proposition.REACH_ZONE_B: crashed_pred,
        Proposition.AVOID_ZONE_A: lambda s: not landed_pred(s),
        Proposition.AVOID_ZONE_B: lambda s: not crashed_pred(s),
    }
    
    # 3. Dummy Agent & Task Composition
    # We won't train a full agent here, just verify the pipeline works
    agent = WVFMultiGoalAgent(terminal_regions=terminals, tasks=tasks)
    
    formula_str = "(!reach_zone_b) U reach_zone_a" # Don't crash until landed
    print(f"\nComposing LTLf Task: {formula_str}")
    ltlf_task = LTLfTask(formula_str, agent)
    
    tracker = DFATracker(ltlf_task.nx_graph, agent.tasks)
    
    # 4. Simulation Loop (Random Actions)
    print(f"\nStarting simulation from DFA State={tracker.current_state}")
    
    max_steps = 100
    steps = 0
    frames = []
    
    # Capture initial frame
    frame = raw_env.render()
    if frame is not None:
        frames.append(Image.fromarray(frame))
    
    while steps < max_steps:
        # Take random action from Gym's action space (0 to 3)
        # LunarLander-v3 discrete actions: 0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine
        action = Action(env.gym_env.action_space.sample())
        
        # Step env
        res = env.step(current_state, action)
        current_state = res.next_state
        steps += 1
        
        # Capture frame
        frame = raw_env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        
        # Step DFA
        old_dfa_state = tracker.current_state
        new_dfa_state = tracker.step_dfa(current_state)
        
        if old_dfa_state != new_dfa_state:
            print(f"DFA Transition: {old_dfa_state} -> {new_dfa_state} at step {steps}")
            
        if res.is_terminal:
            print(f"Environment reached a terminal state at step {steps}.")
            break
            
    print(f"Simulation finished at step {steps}. Final DFA State: {tracker.current_state}")
    
    # 5. Save Results
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, "lunarlander_random.gif")
    
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=33, # roughly 30fps
            loop=0
        )
        print(f"Simulation GIF saved to {output_path}")

if __name__ == "__main__":
    test_masa_lunarlander()