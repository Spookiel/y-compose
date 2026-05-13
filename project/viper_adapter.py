import numpy as np
import torch
import random
from reach_avoid_tabular import Room
from viper.core.dt import DTPolicy
from boolean_task import GoalOrientedQLearning

class RoomGymWrapper:
    """Wraps the custom Room environment to follow the OpenAI Gym interface used by VIPER."""
    def __init__(self, room: Room, goal_mask: torch.Tensor, obstacle_mask: torch.Tensor, max_steps: int = 200):
        self.room = room
        self.goal_mask = goal_mask
        self.obstacle_mask = obstacle_mask
        self.unwrapped = room 
        self.max_steps = max_steps
        self.curr_step = 0

    def reset(self, start_state=None):
        # Room.start returns the initial location as np.ndarray
        self.curr_step = 0
        return self.room.start(start_state=start_state)

    def step(self, action):
        self.curr_step += 1
        # Room.step returns (new_loc, label)
        new_loc, label = self.room.step(action)
        obs = np.array(new_loc)
        
        # Calculate reward and termination for a specific reach-avoid subgoal
        done = False
        reward = 0.0
        if self.goal_mask[tuple(new_loc)]:
            reward = 100.0
            done = True
        elif self.obstacle_mask[tuple(new_loc)] or label == 0:
            # Penalize but do not terminate, matching teacher training logic
            reward = -10.0
            done = False
        if self.curr_step >= self.max_steps:
            done = True
        
        return obs, reward, done, {"label": label}

    def render(self):
        pass

class VIPERTeacher:
    """Wraps GoalOrientedQLearning to provide the batched predict/predict_q methods for DAgger."""
    def __init__(self, qmodel: GoalOrientedQLearning, gr_index: int):
        self.qmodel = qmodel
        self.gr = gr_index

    def predict(self, obss):
        # obss is (N, state_dim)
        # Ensure select_action only picks from valid actions in the wrapped room
        return np.array([self.qmodel.select_action(self.qmodel.Q_subgoal, 
                         obs.astype(int), self.gr) for obs in obss])

    def predict_q(self, obss):
        # Returns Q values for all actions for each state
        # Slice the Q-table to only return the actions available to the room
        return np.array([self.qmodel.Q_subgoal[tuple(obs.astype(int)) + (self.gr,)][:self.qmodel.env.n_actions].numpy() for obs in obss])

class EpsilonGreedyDTPolicy(DTPolicy):
    """A wrapper for DTPolicy that adds epsilon-greedy exploration during distillation rollouts."""
    def __init__(self, max_depth, epsilon=0.1, n_actions=4):
        super().__init__(max_depth)
        self.epsilon = epsilon
        self.n_actions = n_actions

    def predict(self, obss):
        # obss is (N, state_dim)
        # If the tree hasn't been trained yet (e.g. initial rollouts), return random actions
        if not hasattr(self, 'tree'):
            return np.random.randint(0, self.n_actions, size=len(obss))
            
        preds = super().predict(obss)
        if self.epsilon > 0:
            # Apply epsilon-greedy exploration to each observation in the batch
            for i in range(len(preds)):
                if random.random() < self.epsilon:
                    preds[i] = random.randint(0, self.n_actions - 1)
        return preds

    def clone(self):
        # When cloning for evaluation, we return a deterministic version (epsilon=0).
        # This ensures that identify_best_policy evaluates the actual tree performance.
        clone = EpsilonGreedyDTPolicy(self.max_depth, epsilon=0, n_actions=self.n_actions)
        if hasattr(self, 'tree'):
            clone.tree = self.tree
        return clone