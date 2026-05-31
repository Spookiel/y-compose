import random
from extension.base import State, Action, StepResult
from extension.env import Env, GridWorldEnv
from extension.policy import WVFMultiGoalAgent

def train_goal_oriented(
    env: Env, 
    agent: WVFMultiGoalAgent, 
    episodes: int, 
    max_steps: int = 100,
    verbose: bool = True
) -> None:
    """
    Implements the training loop from Algorithm 1: Goal-oriented Q-learning.
    Uses the agent's exploration strategy to learn across all predefined goals simultaneously.
    """
    for episode in range(episodes):
        # 1. Initialize state (Randomly within bounds if GridWorldEnv)
        if isinstance(env, GridWorldEnv):
            state = State(
                x=random.randint(env.x_min, env.x_max),
                y=random.randint(env.y_min, env.y_max)
            )
        else:
            state = State(x=0, y=0)
            
        steps = 0
        total_reward = 0.0
        
        # 2. Episode loop
        while steps < max_steps:
            # Select action using the goal-oriented exploration strategy
            action = agent.get_exploration_action(state)
            
            # Take action
            result = env.step(state, action)
            
            # -- Symbolic Bridge: Determine Terminality --
            # In this framework, the environment is a pure physics engine.
            # Terminality is defined by the agent's goal regions.
            is_goal_terminal = any(region.contains(result.next_state) for region in agent.terminal_regions)
            
            # If it's a goal state, we override terminality in the result passed to the agent
            if is_goal_terminal:
                result = StepResult(
                    next_state=result.next_state,
                    base_reward=result.base_reward,
                    is_terminal=True
                )
            
            # Update all Q-tables simultaneously (Algorithm 1 implementation)
            agent.train_off_policy(state, action, result)
            
            # Transition
            total_reward += result.base_reward
            state = result.next_state
            steps += 1
            
            if result.is_terminal:
                break
                
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes} completed in {steps} steps. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Quick sanity check if run directly
    from extension.goal import GoalRegion
    from extension.proposition import Proposition
    
    test_env = GridWorldEnv(0, 5, 0, 5)
    goals = [GoalRegion(Proposition.REACH_ZONE_A, lambda s: s.x == 5 and s.y == 5)]
    test_agent = WVFMultiGoalAgent(goals)
    
    print("Starting sample training...")
    train_goal_oriented(test_env, test_agent, episodes=100)
    print("Training complete.")
