from __future__ import annotations

import json
from pathlib import Path

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy
from src.harness.trainer import train_agent
from src.policies import BatteryThresholdPolicy, DwellGreedyPolicy, LearnedPolicy, MaxFeasiblePolicy, NoChargingPolicy, UniformPolicy


def build_policy(method: str, env: EBusDroneEnv | None = None):
    if method == "no_charging": return NoChargingPolicy()
    if method.startswith("uniform_"): return UniformPolicy(int(method.split("_")[1]))
    if method == "max_feasible": return MaxFeasiblePolicy()
    if method == "dwell_greedy": return DwellGreedyPolicy()
    if method == "battery_threshold": return BatteryThresholdPolicy()
    train_env = env if env is not None else EBusDroneEnv(smoke_test=True)
    agent = train_agent(train_env, method=method, episodes=2, max_steps=10)
    return LearnedPolicy(agent)


def run_benchmark(methods, out_path: str, env: EBusDroneEnv | None = None, smoke_test: bool = False):
    eval_env = env if env is not None else EBusDroneEnv(smoke_test=smoke_test)
    results = {}
    for m in methods:
        policy = build_policy(m, eval_env)
        results[m] = evaluate_policy(eval_env, policy, episodes=1, max_steps=20)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
