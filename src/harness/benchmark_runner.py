from __future__ import annotations
import csv, json, time
from pathlib import Path
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy
from src.harness.trainer import train_agent
from src.harness.result_aggregator import aggregate
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.policies import BatteryThresholdPolicy, DwellGreedyPolicy, LearnedPolicy, MaxFeasiblePolicy, NoChargingPolicy, UniformPolicy

AGENT_MAP={"dqn_dr":DQNDRAgent,"ddqn_dr":DDQNDRAgent,"am_ddqn_dr":AMDDQNDRAgent,"proposed":AMDuelingDDQNDRAgent,"am_dueling_ddqn_dr":AMDuelingDDQNDRAgent}

def normalize_method_name(method:str)->str:
    if method == 'dwell_based_greedy':
        return 'dwell_greedy'
    if method == 'proposed':
        return 'am_dueling_ddqn_dr'
    return method


def uniform_seconds_from_method(method: str) -> int:
    method = normalize_method_name(method)
    if not method.startswith('uniform_'):
        raise ValueError(f"Method is not a uniform charging policy: {method}")
    suffix = method.removeprefix('uniform_')
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError(f"Invalid uniform method format: {method}. Expected uniform_<positive_integer_seconds>.")
    return int(suffix)

def build_policy(method: str, env: EBusDroneEnv, out_root='outputs', checkpoint: str|None=None, train_if_missing: bool=False, smoke_test: bool=False, cfg:dict|None=None, seed:int=0, instance_name:str='unknown'):
    method=normalize_method_name(method)
    if method == 'no_charging': return NoChargingPolicy()
    if method.startswith('uniform_'): return UniformPolicy(uniform_seconds_from_method(method))
    if method == 'max_feasible': return MaxFeasiblePolicy()
    if method == 'dwell_greedy': return DwellGreedyPolicy()
    if method == 'battery_threshold': return BatteryThresholdPolicy()
    if method not in AGENT_MAP: raise ValueError(f'Unknown method: {method}')
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
    methods=[normalize_method_name(m) for m in methods]
    rows=[]
    for seed in test_seeds:
        for m in methods:
            env = env_builder(seed)
            t0=time.time()
            pol = build_policy(m, env, out_root=cfg['paths']['outputs'], train_if_missing=train_if_missing, smoke_test=smoke_test, cfg=cfg, seed=seed, instance_name=instance_name)
            met=evaluate_policy(env, pol, episodes=1, max_steps=10 if smoke_test else None)
            met.update({'method':m,'instance':instance_name,'seed':seed,'runtime_sec':time.time()-t0,'smoke':bool(smoke_test),'smoke_mode':bool(smoke_test)})
            rows.append(met)
    if not rows: raise ValueError('Benchmark produced no rows.')
    for m in methods:
        if not any(r['method']==m for r in rows): raise ValueError(f'No results for method: {m}')
    p=Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg={m:aggregate([r for r in rows if r['method']==m]) for m in methods}
    Path(out_csv).with_suffix('.json').write_text(json.dumps({'metadata':{'instance':instance_name,'seeds':test_seeds,'methods':methods},'aggregated':agg}, indent=2), encoding='utf-8')
    return rows
