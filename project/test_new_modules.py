import unittest
import torch

from wvf_composition import generate_reward_function, wvf_pointwise_max, wvf_pointwise_min
from product_mdp import ProductMDPGenerator
from branch_and_bound_planner import BranchAndBoundPlanner

class MockEnv:
    def __init__(self):
        self.shape = (5, 5)
        self.n_actions = 4
        self.action_map = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        self.goals = {"g1": torch.zeros((5,5))}
        self.goals["g1"][1,1] = 1

class MockProductMDP:
    def __init__(self):
        self.accepting_states = {2}
    
    def get_successors(self, s, q):
        if s == 0 and q == 0:
            yield (1, 1, 1.0)
            yield (2, 2, 10.0)
        elif s == 1 and q == 1:
            yield (2, 2, 1.0)

class TestWVFComposition(unittest.TestCase):
    def test_reward_fn(self):
        r_fn = generate_reward_function("edge", R_MAX=10.0, r_MIN=-100.0)
        self.assertEqual(r_fn(None, True, False), 10.0)
        self.assertEqual(r_fn(None, False, True), -100.0)
        self.assertEqual(r_fn(None, False, False), -1.0)

    def test_pointwise_max_min(self):
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[0.0, 5.0], [2.0, 6.0]])
        
        max_t = wvf_pointwise_max([t1, t2])
        self.assertTrue(torch.equal(max_t, torch.tensor([[1.0, 5.0], [3.0, 6.0]])))
        
        min_t = wvf_pointwise_min([t1, t2])
        self.assertTrue(torch.equal(min_t, torch.tensor([[0.0, 2.0], [2.0, 4.0]])))

class TestProductMDP(unittest.TestCase):
    def test_get_successors(self):
        env = MockEnv()
        dfa_matrix = {0: {(): 0, (('g1', 1.0),): 1}}
        mdp = ProductMDPGenerator(env, dfa_matrix, {1}, {2})
        
        successors = mdp.get_successors((1, 0), 0)
        # 4 actions, state should transition, one of the transitions should go to (1,1) where g1 is true
        self.assertEqual(len(successors), 4)
        found_q1 = False
        for s_prime, q_prime, cost in successors:
            if s_prime == (1, 1) and q_prime == 1:
                found_q1 = True
        self.assertTrue(found_q1)

class TestPlanner(unittest.TestCase):
    def test_plan(self):
        mdp = MockProductMDP()
        planner = BranchAndBoundPlanner(mdp)
        path, cost = planner.plan(0, 0, lambda s: 0.0)
        self.assertEqual(cost, 2.0)

if __name__ == '__main__':
    unittest.main()
