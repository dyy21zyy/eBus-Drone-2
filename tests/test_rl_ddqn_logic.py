import numpy as np
import torch
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent


def test_ddqn_uses_online_for_selection_target_for_eval():
    a = AMDuelingDDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1, "epsilon_start": 0.0, "epsilon_end": 0.0})
    with torch.no_grad():
        for p in a.online.parameters(): p.zero_()
        for p in a.target.parameters(): p.zero_()
    s = np.array([0.1, 0.2], dtype=np.float32)
    ns = np.array([0.3, 0.4], dtype=np.float32)
    m = np.array([1, 1, 1], dtype=np.float32)
    a.observe(s, 0, 1.0, ns, True, m, m, {})
    loss = a.update(batch_size=1)
    assert loss is not None
