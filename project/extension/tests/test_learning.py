import os
import sys
import pytest
from extension.base import State, Action
from extension.env import GridWorldEnv
from extension.goal import GoalRegion
from extension.proposition import Proposition
from extension.policy import WVFMultiGoalAgent
from extension.learning import train_goal_oriented
from extension.algebra import NegatedQFunction, xor_q_functions


def test_goal_oriented_learning():
    """
    Verifies that the Goal-Oriented Q-learning loop runs and the agent 
    shows some learning progress on an 8x8 grid.
    """
    # 1. Setup Environment (8x8 grid)
    env = GridWorldEnv(x_min=0, x_max=7, y_min=0, y_max=7)
    
    # 2. Define Goals
    # Goal A at x=3, acting as a partial wall
    goal_a = GoalRegion(Proposition.REACH_ZONE_A, lambda s: s.x == 3 and s.y <= 6)
    # Goal B at bottom-right (7,0)
    goal_b = GoalRegion(Proposition.REACH_ZONE_B, lambda s: s.x == 7 and s.y == 0)
    
    agent = WVFMultiGoalAgent(defined_goals=[goal_a, goal_b], epsilon=0.1)
    
    # 3. Train
    print("\nTraining agent on 8x8 grid with wall...")
    # Increase episodes to handle the wall/8x8 space
    train_goal_oriented(env, agent, episodes=400000, verbose=False)
    
    # 4. Verification: Q-values should be non-zero
    q_values = agent.q_tables[Proposition.REACH_ZONE_B].values()
    assert any(q != 0 for q in q_values), "Q-table for Goal B is still all zeros!"
    
    # 5. Visualization
    from extension.render import GridWorldRenderer
    renderer = GridWorldRenderer(env, [goal_a, goal_b])
    
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    
    # Get base Q-functions
    q_max = agent.get_q_function(Proposition.WVF_MAX)
    q_min = agent.get_q_function(Proposition.WVF_MIN)
    q_a = agent.get_q_function(Proposition.REACH_ZONE_A)
    q_b = agent.get_q_function(Proposition.REACH_ZONE_B)
    
    # Compositions
    not_a = NegatedQFunction(q_a, q_max, q_min)
    a_or_b = q_a | q_b
    a_and_b = q_a & q_b
    a_xor_b = xor_q_functions(q_a, q_b, q_max, q_min)
    
    print("Visualizing WVFs and Boolean Task Algebra...")
    renderer.render_value_function(q_max, "WVF_MAX", save_path=os.path.join(static_dir, "wvf_max.png"))
    renderer.render_value_function(q_min, "WVF_MIN", save_path=os.path.join(static_dir, "wvf_min.png"))
    renderer.render_value_function(q_a, "Goal A (Wall)", save_path=os.path.join(static_dir, "wvf_goal_a.png"))
    renderer.render_value_function(q_b, "Goal B", save_path=os.path.join(static_dir, "wvf_goal_b.png"))
    
    renderer.render_value_function(not_a, "NOT Goal A", save_path=os.path.join(static_dir, "wvf_not_a.png"))
    renderer.render_value_function(a_or_b, "Goal A OR Goal B", save_path=os.path.join(static_dir, "wvf_or.png"))
    renderer.render_value_function(a_and_b, "Goal A AND Goal B", save_path=os.path.join(static_dir, "wvf_and.png"))
    renderer.render_value_function(a_xor_b, "Goal A XOR Goal B", save_path=os.path.join(static_dir, "wvf_xor.png"))
    
    print("All visualizations saved to static/")

if __name__ == "__main__":
    test_goal_oriented_learning()
