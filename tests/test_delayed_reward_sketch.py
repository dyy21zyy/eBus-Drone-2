import csv
import numpy as np

from src.harness.trainer import train_agent
from src.rl.replay_buffer import ReplayBuffer
from src.rl.sketch_buffer import SketchBuffer


class _OneStepTerminalEnv:
    def __init__(self, reward=-7.5):
        self.reward = reward
        self.horizon_sec = 60.0
        self.state = {"time": 0.0}
        self.called = 0

    def reset(self, seed=None):
        self.called = 0
        self.state = {"time": 0.0}
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def get_action_mask(self):
        return np.array([1.0, 1.0], dtype=np.float32)

    def step(self, _action):
        self.called += 1
        self.state["time"] = 60.0
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            float(self.reward),
            True,
            False,
            {"termination_reason": "horizon_reached", "reward_components": {"terminal_undelivered_penalty": abs(self.reward)}},
        )


def test_sketch_start_finalize_creates_single_completed_transition():
    sb = SketchBuffer()
    sb.start(np.array([1.0, 2.0]), 1, np.array([1.0, 0.0]), {"decision_event": 1})
    out = sb.finalize(3.0, np.array([2.0, 3.0]), False, np.array([1.0, 1.0]), {"executed_action": 1})
    assert out is not None
    assert out["reward"] == 3.0
    assert out["done"] is False
    assert sb.has_pending() is False


def test_replay_buffer_receives_only_completed_transitions():
    rb = ReplayBuffer(10)
    sb = SketchBuffer()
    sb.start(np.array([0.0, 0.0]), 0, np.array([1.0, 1.0]), {})
    assert len(rb) == 0
    completed = sb.finalize(1.0, np.array([1.0, 1.0]), False, np.array([1.0, 1.0]), {})
    rb.add(**completed)
    assert len(rb) == 1


def test_training_episode_emits_delayed_reward_metrics_and_terminal_transition(tmp_path):
    env = _OneStepTerminalEnv(reward=-11.0)
    train_agent(
        env,
        method="proposed",
        episodes=1,
        smoke_test=False,
        out_root=str(tmp_path),
        cfg={"rl": {"episodes": 1}},
        seed=1,
        instance_name="small",
    )
    with (tmp_path / "metrics" / "train_log_proposed_small_seed_1.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["delayed_reward_sketch_count"]) == 1
    assert int(row["completed_transition_count"]) == 1
    assert int(row["incomplete_sketch_count"]) == 0
    assert int(row["terminal_transition_count"]) == 1
    assert int(row["replay_insertions_episode"]) == 1
    assert float(row["episode_reward"]) == -11.0
