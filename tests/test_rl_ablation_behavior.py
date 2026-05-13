import numpy as np

from src.env.ebus_drone_env import EBusDroneEnv
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent


def test_am_ddqn_never_selects_infeasible_actions():
    agent = AMDDQNDRAgent(3, 5, {"epsilon_start": 1.0, "epsilon_end": 1.0})
    obs = np.zeros(3, dtype=np.float32)
    mask = np.array([1, 0, 0, 1, 0], dtype=np.float32)
    for _ in range(100):
        action = agent.select_action(obs, mask, training=True)
        assert mask[action] == 1


def test_environment_step_info_contract_mentions_action_repair_flag():
    text = open("src/env/ebus_drone_env.py", "r", encoding="utf-8").read()
    assert '"action_repaired": ex_idx != action_index' in text
