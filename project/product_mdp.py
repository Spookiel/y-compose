import numpy as np

class ProductMDPGenerator:
    """
    Implicitly generates the \mathcal{S} \times \mathcal{Q}_{DFA} state space on the fly.
    """
    def __init__(self, env, dfa_matrix, accepting_states, rejecting_states):
        self.env = env
        self.dfa_matrix = dfa_matrix
        self.accepting_states = accepting_states
        self.rejecting_states = rejecting_states

    def infer_next_state(self, s, a, wvf=None):
        """
        Calculates the Bellman Mean Squared Error (MSE) over the local 3x3 grid neighborhood
        to deterministically infer the next physical state s' from the WVF.
        """
        best_s_prime = s
        min_mse = float('inf')
        r, c = s

        # 3x3 neighborhood search
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                s_prime = (r + dr, c + dc)
                if 0 <= s_prime[0] < self.env.shape[0] and 0 <= s_prime[1] < self.env.shape[1]:
                    if wvf is not None:
                        val_s = wvf[r, c]
                        val_s_prime = wvf[s_prime[0], s_prime[1]]
                        # Example Bellman check (gamma=0.99, penalty=-1)
                        mse = (val_s - (-1 + 0.99 * val_s_prime))**2
                        if mse < min_mse:
                            min_mse = mse
                            best_s_prime = s_prime
        return best_s_prime

    def get_successors(self, s, q, wvf=None):
        """
        Computes the next physical state s', evaluates the atomic propositions at s', 
        transitions the DFA to q', and yields valid (s', q', cost) tuples.
        Filters out any transitions to rejecting DFA states.
        """
        successors = []
        for a in range(self.env.n_actions):
            if wvf is not None:
                s_prime = self.infer_next_state(s, a, wvf)
            else:
                move = self.env.action_map[a]
                s_prime = (
                    max(0, min(self.env.shape[0]-1, s[0] + move[0])),
                    max(0, min(self.env.shape[1]-1, s[1] + move[1]))
                )
            
            # Evaluate atomic propositions at s'
            props = tuple()
            if hasattr(self.env, "goals"):
                props_dict = {name: goal[s_prime[0], s_prime[1]].item() for name, goal in self.env.goals.items()}
                # Abstract proposition mapping
                props = tuple(sorted(props_dict.items()))
                
            # Transition the DFA
            q_prime = q 
            if isinstance(self.dfa_matrix, dict):
                q_prime = self.dfa_matrix.get(q, {}).get(props, q)

            if q_prime in self.rejecting_states:
                continue
                
            cost = 1.0 # Uniform transition cost per step
            successors.append((s_prime, q_prime, cost))
            
        return successors
