import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from z3 import *

# 1. HELPER: Convert Scikit-Learn Tree to Z3 Logic
def tree_to_z3(clf, features, node_id=0):
    """
    Recursively converts a sklearn tree into Z3 logic.
    Returns the actual class value from clf.classes_ instead of the internal index.
    """
    tree_ = clf.tree_
    if tree_.feature[node_id] != -2:  # Decision node
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        
        left = tree_to_z3(clf, features, tree_.children_left[node_id])
        right = tree_to_z3(clf, features, tree_.children_right[node_id])
        return If(features[feature_idx] <= threshold, left, right)
    else:
        # Get index of highest probability class
        class_idx = np.argmax(tree_.value[node_id][0])
        # Return the actual value assigned during training (e.g., 3 for Right)
        return int(clf.classes_[class_idx])

def Min(a, b):
    return If(a <= b, a, b)
def Max(a,b):
    return If(a <= b, b, a)
# 2. ENVIRONMENT DYNAMICS (Gridworld)
def step_dynamics(x, y, action, x_next, y_next):
    """Encodes gridworld movement: 0=Up, 1=Down, 2=Left, 3=Right."""
    return And(
        Implies(action == 0, And(x_next == x, y_next == Min(4, y + 1))),
        Implies(action == 1, And(x_next == x, y_next == Max(0, y - 1))),
        Implies(action == 2, And(x_next == Max(0, x - 1), y_next == y)),
        Implies(action == 3, And(x_next == Min(4, x + 1), y_next == y)),
    )

# 3. VERIFICATION SCRIPT
def verify_policy(clf, horizon=2):
    solver = Solver()
    
    # State variables for each timestep
    X = [Int(f'x_{t}') for t in range(horizon + 1)]
    Y = [Int(f'y_{t}') for t in range(horizon + 1)]
    Q = [Int(f'q_{t}') for t in range(horizon + 1)] # DFA state
    A = [Int(f'a_{t}') for t in range(horizon)]     # Action taken

    # Initial State (Start at 0,0; DFA state 0)
    solver.add(X[0] == 0, Y[0] == 0, Q[0] == 0)

    # Goal and Obstacle positions
    GOAL = (2, 0)
    OBSTACLE = (1, 0)

    for t in range(horizon):
        # A. Apply Decision Tree Policy
        # We pass [X_t, Y_t] to the tree translator
        policy_expr = tree_to_z3(clf, [X[t], Y[t]])
        solver.add(A[t] == policy_expr)
        print(policy_expr)
        # B. Apply Dynamics
        s_d = step_dynamics(X[t], Y[t], A[t], X[t+1], Y[t+1])
        solver.add(s_d)

        # C. DFA Transitions (Encoded LTL: F(Goal) & G(!Obstacle))
        # q=0: searching, q=1: goal reached, q=-1: failed (hit obstacle)
        is_at_goal = And(X[t+1] == GOAL[0], Y[t+1] == GOAL[1])
        is_at_obs = And(X[t+1] == OBSTACLE[0], Y[t+1] == OBSTACLE[1])
        print(is_at_goal)
        dfa_transition = If(is_at_obs, Q[t+1] == -1,
                         If(And(Q[t] == 0, is_at_goal), Q[t+1] == 1,
                         Q[t+1] == Q[t]))
        solver.add(dfa_transition)

    # 4. DEFINE THE VIOLATION (Is there a path that hits obstacle OR never reaches goal?)
    # Counter-example: Never reach state 1 OR hit state -1
    property_violated = Or(
        [Q[t] == -1 for t in range(horizon + 1)] + # Safety violation
        [Q[horizon] != 1]                          # Liveness violation (didn't reach goal)
    )
    solver.add(property_violated)

    if solver.check() == sat:
        print("❌ Verification Failed! Counter-example found:")
        m = solver.model()
        for t in range(horizon + 1):
            print(f"T={t}: Pos=({m[X[t]]}, {m[Y[t]]}), DFA_State={m[Q[t]]}")
    else:
        print("✅ Policy Verified! No violations found within horizon.")

# --- MOCK DATA FOR TESTING ---
# Creating a simple tree that mostly goes Right (3) then Up (0)
X_train = np.array([[0,0], [1,0], [2,0], [3,0], [4,0], [4,1], [4,2], [4,3]])
y_train = np.array([3, 3, 3, 3, 3, 0, 0, 0])
mock_clf = DecisionTreeClassifier().fit(X_train, y_train)
print(export_text(mock_clf))

verify_policy(mock_clf, horizon=2)