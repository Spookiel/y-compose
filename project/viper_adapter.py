import numpy as np
import torch
import random
from reach_avoid_tabular import Room
from viper.core.dt import DTPolicy
from boolean_task import GoalOrientedQLearning

class ActionMapper:
    """
    Handles mapping between VIPER student actions and Room environment actions.
    Ensures consistent action indexing between y-compose and the extracted policy.
    """
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        # Default Mapping for 4 actions:
        # Room indices (y-compose): 0:North, 1:East, 2:South, 3:West
        # VIPER student indices: 0:West(Left), 1:South(Down), 2:East(Right), 3:North(Up)
        if n_actions == 4:
            self.room_to_viper = {0: 3, 1: 2, 2: 1, 3: 0}
            self.viper_to_room = {3: 0, 2: 1, 1: 2, 0: 3}
        else:
            # Identity mapping for other configurations (e.g. 8 actions)
            self.room_to_viper = {i: i for i in range(n_actions)}
            self.viper_to_room = {i: i for i in range(n_actions)}

    def to_room(self, viper_action):
        return self.viper_to_room.get(int(viper_action), int(viper_action))

    def to_viper(self, room_action):
        return self.room_to_viper.get(int(room_action), int(room_action))

    def reorder_q(self, q_values):
        """Reorders Q-values array from Room index ordering to VIPER index ordering."""
        reordered = np.zeros_like(q_values)
        for r_idx, v_idx in self.room_to_viper.items():
            if r_idx < len(q_values) and v_idx < len(reordered):
                reordered[v_idx] = q_values[r_idx]
        return reordered

class RoomGymWrapper:
    """Wraps the custom Room environment to follow the OpenAI Gym interface used by VIPER."""
    def __init__(self, room: Room, goal_mask: torch.Tensor, obstacle_mask: torch.Tensor, max_steps: int = 200, mapper=None):
        self.room = room
        self.goal_mask = goal_mask
        self.obstacle_mask = obstacle_mask
        self.unwrapped = room 
        self.max_steps = max_steps
        self.curr_step = 0
        self.mapper = mapper or ActionMapper(room.n_actions)

    def reset(self, start_state=None):
        # Room.start returns the initial location as np.ndarray
        self.curr_step = 0
        return self.room.start(start_state=start_state)

    def step(self, action):
        self.curr_step += 1
        # Map student action to Room action index
        mapped_action = self.mapper.to_room(action)
        
        # Room.step returns (new_loc, label)
        new_loc, label = self.room.step(mapped_action)
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
    def __init__(self, qmodel: GoalOrientedQLearning, gr_index: int, mapper=None):
        self.qmodel = qmodel
        self.gr = gr_index
        self.mapper = mapper or ActionMapper(qmodel.env.n_actions)

    def predict(self, obss):
        # obss is (N, state_dim)
        # Use the argmax of the Q-table to get the teacher's expert action, then map to student space.
        # We manually find the best action to ensure a random choice among ties without teacher epsilon.
        actions = []
        for obs in obss:
            q_vals = self.qmodel.Q_subgoal[tuple(obs.astype(int)) + (self.gr,)][:self.qmodel.env.n_actions]
            max_q = q_vals.max()
            best_actions = (q_vals == max_q).nonzero(as_tuple=True)[0]
            room_action = random.choice(best_actions).item()
            actions.append(self.mapper.to_viper(room_action))
        return np.array(actions)

    def predict_q(self, obss):
        # Returns Q values for all actions for each state
        # Slice the Q-table to only return the actions available to the room
        # and reorder them to match the student's action mapping.
        q_batch = []
        for obs in obss:
            q_vals = self.qmodel.Q_subgoal[tuple(obs.astype(int)) + (self.gr,)][:self.qmodel.env.n_actions].numpy()
            q_batch.append(self.mapper.reorder_q(q_vals))
        return np.array(q_batch)

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
            
        preds = np.copy(super().predict(obss))
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