import numpy as np

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.trainer import train_agent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.replay_buffer import ReplayBuffer


def test_replay_buffer_stores_realized_reward():
    b = ReplayBuffer(10)
    b.add(np.zeros(3), 1, -3.5, np.ones(3), False, np.array([1, 0]), np.array([1, 1]), {"x": 1})
    assert b.sample(1)[0].reward == -3.5


def test_checkpoint_roundtrip_greedy_action(tmp_path):
    cfg = {"epsilon_start": 0.0, "epsilon_end": 0.0}
    a1 = AMDuelingDDQNDRAgent(4, 5, cfg)
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    p = tmp_path / "a.pt"
    a1.save_checkpoint(str(p))
    a2 = AMDuelingDDQNDRAgent(4, 5, cfg)
    a2.load_checkpoint(str(p))
    assert a1.select_action(obs, mask, training=False) == a2.select_action(obs, mask, training=False)


def test_training_smoke_writes_checkpoint_and_log(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    train_agent(env, method="proposed", episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg={"rl": {"episodes": 1, "max_steps_per_episode": 2}})
    assert (tmp_path / "checkpoints" / "proposed.pt").exists()
    assert (tmp_path / "metrics" / "train_log_proposed.csv").exists()
