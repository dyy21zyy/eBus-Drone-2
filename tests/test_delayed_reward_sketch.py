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
    with (tmp_path / "metrics" / "train_log_am_dueling_ddqn_dr_small_seed_1.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["delayed_reward_sketch_count"]) == 1
    assert int(row["completed_transition_count"]) == 1
    assert int(row["incomplete_sketch_count"]) == 0
    assert int(row["terminal_transition_count"]) == 1
    assert int(row["replay_insertions_episode"]) == 1
    assert float(row["episode_reward"]) == -11.0


def test_replay_buffer_total_added_counts_insertions_after_capacity():
    rb = ReplayBuffer(capacity=3)
    for i in range(5):
        rb.add(
            observation=np.array([float(i)], dtype=np.float32),
            action_index=0,
            reward=1.0,
            next_observation=np.array([float(i + 1)], dtype=np.float32),
            done=False,
            action_mask=np.array([1.0], dtype=np.float32),
            next_action_mask=np.array([1.0], dtype=np.float32),
            info={},
        )
    assert len(rb) == 3
    assert rb.total_added == 5


class _MultiStepTerminalEnv:
    def __init__(self, done_after=8):
        self.done_after = done_after
        self.state = {"time": 0.0}
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        self.state = {"time": 0.0}
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def get_action_mask(self):
        return np.array([1.0, 1.0], dtype=np.float32)

    def step(self, _action):
        self._steps += 1
        self.state["time"] = float(self._steps * 10.0)
        done = self._steps >= self.done_after
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            1.0,
            done,
            False,
            {"termination_reason": "horizon_reached" if done else None, "reward_components": {}},
        )


def test_training_accounting_uses_total_insertions_when_buffer_is_full(tmp_path):
    env = _MultiStepTerminalEnv(done_after=8)
    train_agent(
        env,
        method="proposed",
        episodes=2,
        smoke_test=False,
        out_root=str(tmp_path),
        cfg={"rl": {"episodes": 2, "replay_buffer_size": 5, "batch_size": 2, "warmup_steps": 1, "progress_print_every": 1}},
        seed=2,
        instance_name="small",
    )
    with (tmp_path / "metrics" / "train_log_am_dueling_ddqn_dr_small_seed_2.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert int(rows[0]["replay_insertions_episode"]) == 8
    assert int(rows[1]["replay_insertions_episode"]) == 8
    assert int(rows[1]["completed_transition_count"]) == 8
    assert int(rows[1]["buffer_len"]) == 5
    assert int(rows[1]["buffer_total_added"]) == 16
