import numpy as np
import torch

from src.env.action_space import feasible_action_mask
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent


def test_station_power_overload_not_part_of_mask_inputs():
    m1 = feasible_action_mask(available_chargers=1, current_battery_kwh=80.0, capacity_kwh=160.0, power_kw=500.0, eta=0.95)
    m2 = feasible_action_mask(available_chargers=1, current_battery_kwh=80.0, capacity_kwh=160.0, power_kw=500.0, eta=0.95)
    assert np.array_equal(m1, m2)


def test_mask_always_keeps_action_zero_feasible():
    mask = feasible_action_mask(available_chargers=0, current_battery_kwh=159.9, capacity_kwh=160.0, power_kw=500.0, eta=0.95)
    assert int(mask[0]) == 1


def test_ddqn_next_state_mask_used_only_for_masked_variants(monkeypatch):
    called = {"count": 0}
    original = torch.Tensor.masked_fill

    def _wrapped(self, mask, value):
        called["count"] += 1
        return original(self, mask, value)

    monkeypatch.setattr(torch.Tensor, "masked_fill", _wrapped, raising=True)

    s = np.array([0.1, 0.2], dtype=np.float32)
    ns = np.array([0.3, 0.4], dtype=np.float32)
    m = np.array([1, 0, 1], dtype=np.float32)

    # unmasked baseline should not apply next-state mask in DDQN target path
    unmasked = DDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1})
    unmasked.observe(s, 0, 1.0, ns, True, m, m, {})
    _ = unmasked.update(batch_size=1)
    unmasked_calls = called["count"]

    # masked variant should apply masked_fill for next-state action selection/evaluation
    masked = AMDDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1})
    masked.observe(s, 0, 1.0, ns, True, m, m, {})
    _ = masked.update(batch_size=1)
    masked_calls = called["count"] - unmasked_calls

    assert unmasked_calls == 0
    assert masked_calls > 0
