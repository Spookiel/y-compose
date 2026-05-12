import numpy as np
import torch
from reach_avoid_tabular import Room
from boolean_task import GoalOrientedQLearning

class RoomGymWrapper:
    """Wraps the custom Room environment to follow the OpenAI Gym interface used by VIPER."""
    def __init__(self, room: Room, goal_mask: torch.Tensor, obstacle_mask: torch.Tensor, max_steps: int = 500):
        self.room = room
        self.goal_mask = goal_mask
        self.obstacle_mask = obstacle_mask
        self.unwrapped = room 
        self.max_steps = max_steps
        self.curr_step = 0

    def reset(self):
        # Room.start returns the initial location as np.ndarray
        self.curr_step = 0
        return self.room.start()

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
            reward = -10.0
            done = True
        elif self.curr_step >= self.max_steps:
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