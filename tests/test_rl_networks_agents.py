import numpy as np, torch
from src.rl.networks.dueling_q_network import DuelingQNetwork
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent


def test_dueling_output_shape():
    d = DuelingQNetwork(4, 9)
    x = torch.randn(2, 4)
    m = torch.ones(2, 9)
    assert d(x, m).shape == (2, 9)


def test_random_exploration_only_feasible():
    agent = AMDDQNDRAgent(4, 5, {"epsilon_start": 1.0, "epsilon_end": 1.0})
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    actions = {agent.select_action(obs, mask, training=True) for _ in range(100)}
    assert actions.issubset({0, 2})


def test_infeasible_never_selected_proposed():
    agent = AMDuelingDDQNDRAgent(4, 5, {})
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 0, 1, 0], dtype=np.float32)
    for _ in range(20):
        a = agent.select_action(obs, mask, training=False)
        assert mask[a] == 1
