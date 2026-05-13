from __future__ import annotations

import json
from pathlib import Path

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy
from src.harness.result_aggregator import aggregate
from src.harness.trainer import train_agent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.policies import BatteryThresholdPolicy, DwellGreedyPolicy, LearnedPolicy, MaxFeasiblePolicy, NoChargingPolicy, UniformPolicy


def build_policy(method: str, env: EBusDroneEnv | None = None, smoke_test: bool=False):
    if method == "no_charging": return NoChargingPolicy()
    if method.startswith("uniform_"): return UniformPolicy(int(method.split("_")[1]))
    if method == "max_feasible": return MaxFeasiblePolicy()
    if method == "dwell_greedy": return DwellGreedyPolicy()
    if method == "battery_threshold": return BatteryThresholdPolicy()
    train_env = env if env is not None else EBusDroneEnv(smoke_test=True)
    ckpt = Path('outputs') / 'checkpoints' / f'{method}.pt'
    if method in {'proposed','am_dueling_ddqn_dr'} and ckpt.exists():
        obs,_=train_env.reset(seed=0)
        agent=AMDuelingDDQNDRAgent(len(obs), len(train_env.get_action_mask()), {'device':'auto'})
        agent.load_checkpoint(str(ckpt))
    else:
        agent = train_agent(train_env, method=method, episodes=1 if smoke_test else 5, max_steps=10, smoke_test=smoke_test)
    return LearnedPolicy(agent)


def run_benchmark(methods, out_path: str, env: EBusDroneEnv | None = None, smoke_test: bool = False):
    eval_env = env if env is not None else EBusDroneEnv(smoke_test=smoke_test)
    raw, agg = {}, {}
    for m in methods:
        policy = build_policy(m, eval_env, smoke_test=smoke_test)
        ep_metrics = [evaluate_policy(eval_env, policy, episodes=1, max_steps=20)]
        raw[m]=ep_metrics
        agg[m]=aggregate(ep_metrics)
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    p.with_name(p.stem+"_aggregated.json").write_text(json.dumps(agg, indent=2), encoding='utf-8')
    return raw
