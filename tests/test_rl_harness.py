import numpy as np

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import run_benchmark
from src.harness.trainer import train_agent
from src.policies import NoChargingPolicy, MaxFeasiblePolicy, UniformPolicy, DwellGreedyPolicy, BatteryThresholdPolicy
from src.rl.replay_buffer import ReplayBuffer


def test_policies_return_action_index():
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
    policies = [NoChargingPolicy(), MaxFeasiblePolicy(), UniformPolicy(60), DwellGreedyPolicy(), BatteryThresholdPolicy()]
    for p in policies:
        a = p.select_action(obs, mask, {"T_P_est": 10, "T_F": 12, "E_current": 50, "E_min": 20, "E_max": 100})
        assert isinstance(a, int)


def test_replay_buffer_stores_masks():
    b = ReplayBuffer(10)
    b.add(np.zeros(3), 1, 1.0, np.ones(3), False, np.array([1,0]), np.array([1,1]), {"x": 1})
    t = b.sample(1)[0]
    assert t.action_mask.tolist() == [1.0, 0.0]
    assert t.next_action_mask.tolist() == [1.0, 1.0]


def test_smoke_train_and_benchmark(tmp_path):
    env = EBusDroneEnv()
    agent = train_agent(env, method="proposed", episodes=1, max_steps=2)
    assert len(agent.buffer) > 0
    out = tmp_path / "bench.json"
    res = run_benchmark(["no_charging", "proposed"], str(out))
    assert "proposed" in res
    assert out.exists()
