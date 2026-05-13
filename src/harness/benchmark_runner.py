from __future__ import annotations
import csv, json, time
from pathlib import Path
from copy import deepcopy

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy
from src.harness.trainer import train_agent, LEARNING_METHODS
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.policies import BatteryThresholdPolicy, DwellGreedyPolicy, LearnedPolicy, MaxFeasiblePolicy, NoChargingPolicy, UniformPolicy

AGENT_MAP={"dqn_dr":DQNDRAgent,"ddqn_dr":DDQNDRAgent,"am_ddqn_dr":AMDDQNDRAgent,"proposed":AMDuelingDDQNDRAgent,"am_dueling_ddqn_dr":AMDuelingDDQNDRAgent}

def build_policy(method: str, env: EBusDroneEnv, out_root='outputs', checkpoint: str|None=None, train_if_missing: bool=False, smoke_test: bool=False, cfg:dict|None=None, seed:int=0, instance_name:str='unknown'):
    if method == "no_charging": return NoChargingPolicy()
    if method.startswith("uniform_"): return UniformPolicy(int(method.split("_")[1]))
    if method == "max_feasible": return MaxFeasiblePolicy()
    if method in {"dwell_greedy","dwell_based_greedy"}: return DwellGreedyPolicy()
    if method == "battery_threshold": return BatteryThresholdPolicy()
    ckpt = Path(checkpoint) if checkpoint else Path(out_root)/'checkpoints'/f'{method}_{instance_name}_seed_{seed}.pt'
    if not ckpt.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"Missing checkpoint for learning method '{method}': {ckpt}. Re-run with --train-if-missing.")
        _, path = train_agent(env, method=method, episodes=1 if smoke_test else 20, max_steps=10 if smoke_test else 100, smoke_test=smoke_test, out_root=out_root, cfg=cfg, seed=seed, instance_name=instance_name)
        ckpt = Path(path)
    obs,_=env.reset(seed=seed)
    agent=AGENT_MAP[method](len(obs), len(env.get_action_mask()), {'device':'auto'})
    agent.load_checkpoint(str(ckpt))
    return LearnedPolicy(agent)


def run_benchmark(methods, out_csv: str, env_builder, instance_name:str, test_seeds:list[int], cfg:dict, smoke_test: bool = False, train_if_missing:bool=False):
    rows=[]
    for seed in test_seeds:
        env_seed = env_builder(seed)
        for m in methods:
            env = env_builder(seed)
            t0=time.time()
            pol = build_policy(m, env, out_root=cfg['paths']['outputs'], train_if_missing=train_if_missing, smoke_test=smoke_test, cfg=cfg, seed=seed, instance_name=instance_name)
            met=evaluate_policy(env, pol, episodes=1, max_steps=10 if smoke_test else 50)
            met.update({"method":m,"instance":instance_name,"seed":seed,"runtime_sec":time.time()-t0})
            rows.append(met)
    p=Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows
