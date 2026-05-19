import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PriorityNode:
    priority: float
    cost: float = field(compare=False)
    state: Any = field(compare=False)
    q_state: Any = field(compare=False)
    path: list = field(compare=False)

class BranchAndBoundPlanner:
    """
    Replaces myopic Dijkstra search with a globally optimal temporal planner over the Product MDP.
    """
    def __init__(self, product_mdp):
        self.mdp = product_mdp

    def plan(self, start_s, start_q, wvf_heuristic):
        """
        Uses the zero-shot WVF evaluation as a strict, admissible lower-bound heuristic 
        to guide the search and aggressively prune branches.
        """
        pq = []
        start_h = wvf_heuristic(start_s)
        heapq.heappush(pq, PriorityNode(
            priority=start_h, cost=0.0, state=start_s, q_state=start_q, path=[(start_s, start_q)]
        ))
        
        best_cost = float('inf')
        best_path = None
        visited = {}

        while pq:
            node = heapq.heappop(pq)

            # Aggressively prune branches that exceed the best known complete path cost
            if node.cost + wvf_heuristic(node.state) >= best_cost:
                continue

            state_key = (node.state, node.q_state)
            if state_key in visited and visited[state_key] <= node.cost:
                continue
            visited[state_key] = node.cost

            # Check if accepted by DFA
            if node.q_state in self.mdp.accepting_states:
                if node.cost < best_cost:
                    best_cost = node.cost
                    best_path = node.path
                continue

            # Expand successors
            for s_prime, q_prime, edge_cost in self.mdp.get_successors(node.state, node.q_state):
                g_new = node.cost + edge_cost
                h_new = wvf_heuristic(s_prime)
                
                if g_new + h_new < best_cost:
                    heapq.heappush(pq, PriorityNode(
                        priority=g_new + h_new,
                        cost=g_new,
                        state=s_prime,
                        q_state=q_prime,
                        path=node.path + [(s_prime, q_prime)]
                    ))

        return best_path, best_cost
