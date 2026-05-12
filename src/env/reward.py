from __future__ import annotations


def compute_reward(passenger_delay: float, action_duration: float, undelivered_penalty: float = 0.0) -> float:
    return -0.01 * max(passenger_delay, 0.0) - 0.001 * max(action_duration, 0.0) - max(undelivered_penalty, 0.0)
