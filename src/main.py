from __future__ import annotations
import argparse, json
from pathlib import Path

from src.data_generation.instance_writer import write_instance
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import build_policy, run_benchmark
from src.harness.ablation_runner import run_ablation
from src.harness.sensitivity_runner import run_sensitivity
from src.harness.evaluator import evaluate_policy, save_eval_metrics
from src.harness.trainer import train_agent
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import load_offline_assignment, write_assignment
from src.offline.assignment_solver import solve_assignment
from src.offline.assignment_validator import summarize_assignment_flows
from src.utils.config import load_instance, load_scenario, load_yaml
from src.utils.random_seed import set_seed

def run_generate(cfg, instance_name: str, seed: int):
    instance_cfg = load_yaml(f"configs/instances/{instance_name}.yaml")
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    write_instance(generate_scenario(cfg, instance, seed, 0), out_root, f"scenario_0_seed_{seed}")

def run_offline(cfg, instance_name: str, seed: int):
    instance = load_instance(instance_name, seed)
    data = build_assignment_data(instance)
    result = solve_assignment(data, allow_greedy_fallback=bool(cfg.get("offline", {}).get("allow_greedy_fallback", False)))
    out = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    metadata = {"instance_name": instance_name, "seed": seed, **summarize_assignment_flows(data, result)}
    write_assignment(result, str(out), metadata=metadata)

def build_env(cfg, instance_name: str, seed: int, smoke_test: bool) -> EBusDroneEnv:
    return EBusDroneEnv(config=cfg, instance=load_instance(instance_name, seed), scenario=load_scenario(instance_name, seed), assignment=load_offline_assignment(instance_name, seed), smoke_test=smoke_test)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate', 'offline', 'train', 'eval', 'benchmark', 'ablation', 'sensitivity'])
    ap.add_argument('--config', required=True); ap.add_argument('--instance', default='small'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--method', default='no_charging'); ap.add_argument('--train-seeds', nargs='+', type=int); ap.add_argument('--test-seeds', nargs='+', type=int)
    ap.add_argument('--episodes', type=int); ap.add_argument('--smoke-test', action='store_true'); ap.add_argument('--checkpoint'); ap.add_argument('--train-if-missing', action='store_true')
    ap.add_argument('--factor'); ap.add_argument('--values', nargs='+', type=float)
    args = ap.parse_args(); cfg = load_yaml(args.config)

    if args.mode == 'generate': run_generate(cfg, args.instance, args.seed); return
    if args.mode == 'offline': run_offline(cfg, args.instance, args.seed); return
    if args.mode == 'train':
        for seed in (args.train_seeds or [args.seed]):
            set_seed(seed, deterministic=bool(cfg.get('rl',{}).get('deterministic',False)))
            env = build_env(cfg, args.instance, seed, args.smoke_test)
            train_agent(env, method=args.method, episodes=args.episodes or (2 if args.smoke_test else 20), max_steps=10 if args.smoke_test else 100, smoke_test=args.smoke_test, out_root=cfg['paths']['outputs'], cfg=cfg, seed=seed, instance_name=args.instance)
        return
    if args.mode == 'eval':
        rows=[]
        for seed in (args.test_seeds or [args.seed]):
            env=build_env(cfg, args.instance, seed, args.smoke_test)
            try:
                pol=build_policy(args.method, env=env, out_root=cfg['paths']['outputs'], checkpoint=args.checkpoint, train_if_missing=False, smoke_test=args.smoke_test, cfg=cfg, seed=seed, instance_name=args.instance)
            except TypeError:
                pol=build_policy(args.method)
            m=evaluate_policy(env, pol, episodes=1, max_steps=10 if args.smoke_test else None); m.update({'method':args.method,'instance':args.instance,'seed':seed}); rows.append(m)
        out=Path(cfg['paths']['outputs'])/'metrics'/f'eval_{args.method}_{args.instance}.csv'; save_eval_metrics(rows, str(out)); print(json.dumps(rows)); return
    if args.mode == 'benchmark':
        methods=load_yaml('configs/experiments/benchmark.yaml').get('methods', [])
        out=Path(cfg['paths']['outputs'])/'results'/'benchmark'/args.instance/'summary.csv'
        run_benchmark(methods, str(out), env_builder=lambda sd: build_env(cfg, args.instance, sd, args.smoke_test), instance_name=args.instance, test_seeds=args.test_seeds or [args.seed], cfg=cfg, smoke_test=args.smoke_test, train_if_missing=args.train_if_missing); return
    if args.mode == 'ablation':
        out=Path(cfg['paths']['outputs'])/'results'/'ablation'/args.instance/'summary.csv'
        run_ablation(str(out), env_builder=lambda sd: build_env(cfg, args.instance, sd, args.smoke_test), instance_name=args.instance, test_seeds=args.test_seeds or [args.seed], cfg=cfg, smoke_test=args.smoke_test, train_if_missing=args.train_if_missing); return
    if args.mode == 'sensitivity':
        out=Path(cfg['paths']['outputs'])/'results'/'sensitivity'/args.instance/f'{args.factor}.csv'
        run_sensitivity(['proposed'], str(out), env_builder=lambda sd, c: build_env(c, args.instance, sd, args.smoke_test), instance_name=args.instance, test_seeds=args.test_seeds or [args.seed], cfg=cfg, factor=args.factor, values=args.values or [1.0], smoke_test=args.smoke_test, train_if_missing=args.train_if_missing); return

if __name__=='__main__': main()
