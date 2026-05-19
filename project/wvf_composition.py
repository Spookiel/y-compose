import torch

def generate_reward_function(edge_formula, R_MAX=10.0, r_MIN=-100.0):
    """
    Dynamically generates a task-specific reward function for a given Boolean edge formula.
    Assigns R_MAX to goals satisfying the formula and a massive penalty r_MIN to violating states.
    """
    def reward_fn(state, satisfies_formula: bool, violates_formula: bool):
        if violates_formula:
            return r_MIN
        if satisfies_formula:
            return R_MAX
        return -1.0 # standard step penalty
    return reward_fn

def wvf_pointwise_max(wvfs: list[torch.Tensor]) -> torch.Tensor:
    """
    Implements the pointwise maximum (max) for disjunctions over the pre-trained base WVFs
    to compute \tilde{\overline{V}}_M(s,g).
    """
    return torch.max(torch.stack(wvfs), dim=0).values

def wvf_pointwise_min(wvfs: list[torch.Tensor]) -> torch.Tensor:
    """
    Implements the pointwise minimum (min) for conjunctions over the pre-trained base WVFs.
    """
    return torch.min(torch.stack(wvfs), dim=0).values
