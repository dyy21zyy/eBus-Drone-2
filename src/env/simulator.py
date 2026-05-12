from __future__ import annotations

from .ebus_drone_env import EBusDroneEnv


def make_env() -> EBusDroneEnv:
    return EBusDroneEnv()
