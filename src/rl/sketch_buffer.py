"""Lightweight delayed-transition sketch buffer.

Keeps a pending (s, a, mask) and materializes full replay transitions once reward/next-state arrive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PendingTransition:
    observation: np.ndarray
    action_index: int
    action_mask: np.ndarray
    info: dict[str, Any]


class SketchBuffer:
    def __init__(self):
        self.pending: PendingTransition | None = None

    def start(self, observation, action_index: int, action_mask, info: dict[str, Any] | None = None) -> None:
        self.pending = PendingTransition(
            observation=np.asarray(observation, dtype=np.float32),
            action_index=int(action_index),
            action_mask=np.asarray(action_mask, dtype=np.float32),
            info=info or {},
        )

    def has_pending(self) -> bool:
        return self.pending is not None

    def finalize(self, reward: float, next_observation, done: bool, next_action_mask, info: dict[str, Any] | None = None):
        if self.pending is None:
            return None
        out = {
            "observation": self.pending.observation,
            "action_index": self.pending.action_index,
            "reward": float(reward),
            "next_observation": np.asarray(next_observation, dtype=np.float32),
            "done": bool(done),
            "action_mask": self.pending.action_mask,
            "next_action_mask": np.asarray(next_action_mask, dtype=np.float32),
            "info": {**self.pending.info, **(info or {})},
        }
        self.pending = None
        return out
