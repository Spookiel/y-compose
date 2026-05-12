from z3 import *
import numpy as np

def tree_to_z3(clf, features, node_id=0):
    """Recursively converts a sklearn tree into Z3 logic."""
    tree_ = clf.tree_
    if tree_.feature[node_id] != -2:  # Decision node
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        left = tree_to_z3(clf, features, tree_.children_left[node_id])
        right = tree_to_z3(clf, features, tree_.children_right[node_id])
        return If(features[feature_idx] <= threshold, left, right)
    else:
        # Leaf node: return the predicted class
        class_idx = np.argmax(tree_.value[node_id][0])
        return int(clf.classes_[class_idx])

def room_step_dynamics(x, y, action, x_next, y_next, shape, n_actions):
    """Encodes the action map from Room.action_map into Z3 logic based on n_actions."""
    h, w = shape
    
    # Action vectors based on Room.action_map
    # 0:N(-1,0), 1:E(0,1), 2:S(1,0), 3:W(0,-1), 4:NE(-1,1), 5:SE(1,1), 6:SW(1,-1), 7:NW(-1,-1)
    
    def get_dx(a):
        return If(Or(a == 0, a == 4, a == 7), -1, 
               If(Or(a == 2, a == 5, a == 6), 1, 0))
    
    def get_dy(a):
        return If(Or(a == 1, a == 4, a == 5), 1, 
               If(Or(a == 3, a == 6, a == 7), -1, 0))

    dx = get_dx(action)
    dy = get_dy(action)
    
    potential_x = x + dx
    potential_y = y + dy
    
    # Check boundaries (Room.step behavior: stay put if out of range)
    out_of_range = Or(potential_x < 0, potential_x >= h, 
                      potential_y < 0, potential_y >= w)
    
    # Note: Obstacles are handled in the DFA transition (stay put/violate) 
    # to match the environment's reach-avoid logic.
    
    if n_actions == 4:
        # Only allow actions 0, 1, 2, 3
        return And(action >= 0, action < 4,
                   x_next == If(out_of_range, x, potential_x),
                   y_next == If(out_of_range, y, potential_y))
    elif n_actions == 8:
        return And(action >= 0, action < 8,
                   x_next == If(out_of_range, x, potential_x),
                   y_next == If(out_of_range, y, potential_y))
    else:
        return False # This will make the solver unsat if this path is taken

def verify_subgoal_dt(dt_policy, room_shape, start_pos, goal_coords, obstacle_coords, n_actions, horizon=20):
    """Verifies that the DT policy reaches the goal without hitting obstacles."""
    solver = Solver()
    clf = dt_policy.tree
    
    X = [Int(f'x_{t}') for t in range(horizon + 1)]
    Y = [Int(f'y_{t}') for t in range(horizon + 1)]
    # DFA state tracker: 0=searching, 1=success, -1=violation
    Q = [Int(f'q_{t}') for t in range(horizon + 1)] 
    
    # Constraints: Initial State and Grid Bounds
    solver.add(X[0] == start_pos[0], Y[0] == start_pos[1], Q[0] == 0)
    for t in range(horizon + 1):
        solver.add(X[t] >= 0, X[t] < room_shape[0], Y[t] >= 0, Y[t] < room_shape[1])

    for t in range(horizon):
        action = tree_to_z3(clf, [X[t], Y[t]])
        
        # Apply Dynamics
        solver.add(room_step_dynamics(X[t], Y[t], action, X[t+1], Y[t+1], room_shape, n_actions))
        
        # Evaluate Regions
        is_goal = Or([And(X[t+1] == g[0], Y[t+1] == g[1]) for g in goal_coords])
        is_obs  = Or([And(X[t+1] == o[0], Y[t+1] == o[1]) for o in obstacle_coords])
        
        # DFA Transitions for F(Goal) & G(!Obstacle)
        dfa_trans = If(is_obs, Q[t+1] == -1,
                    If(And(Q[t] == 0, is_goal), Q[t+1] == 1,
                    Q[t+1] == Q[t]))
        solver.add(dfa_trans)

    # Look for a violation: either hit an obstacle or failed to reach the goal by the end
    violation = Or(
        Or([Q[t] == -1 for t in range(horizon + 1)]),
        Q[horizon] != 1
    )
    solver.add(violation)

    if solver.check() == sat:
        print("❌ Specification Violated! Counter-example path:")
        m = solver.model()
        for t in range(horizon + 1):
            print(f"t={t}: ({m[X[t]]}, {m[Y[t]]}) DFA_State={m[Q[t]]}")
        return False
    else:
        print("✅ Policy Verified against Specification!")
        return True