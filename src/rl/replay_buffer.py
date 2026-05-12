from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action_index: int
    reward: float
    next_observation: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    info: dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self.data: deque[Transition] = deque(maxlen=self.capacity)

    def add(
        self,
        observation: np.ndarray,
        action_index: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.data.append(
            Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action_index=int(action_index),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                done=bool(done),
                action_mask=np.asarray(action_mask, dtype=np.float32),
                next_action_mask=np.asarray(next_action_mask, dtype=np.float32),
                info=info or {},
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)
