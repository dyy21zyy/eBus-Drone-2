from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.data_generation.instance_writer import write_instance
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import build_policy, run_benchmark
from src.harness.ablation_runner import run_ablation
from src.harness.sensitivity_runner import run_sensitivity
from src.harness.evaluator import evaluate_policy
from src.harness.trainer import train_agent
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import load_offline_assignment, write_assignment
from src.offline.assignment_solver import solve_assignment
from src.utils.config import load_instance, load_scenario, load_yaml


def run_generate(cfg, instance_name: str, seed: int):
    instance_cfg = load_yaml(f"configs/instances/{instance_name}.yaml")
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    write_instance(generate_scenario(cfg, instance, seed, 0), out_root, f"scenario_0_seed_{seed}")


def run_offline(cfg, instance_name: str, seed: int):
    instance = load_instance(instance_name, seed)
    allow_fallback = bool(cfg.get("offline", {}).get("allow_greedy_fallback", False))
    result = solve_assignment(build_assignment_data(instance), allow_greedy_fallback=allow_fallback)
    print(f"[offline] solver={result.solver_name} status={result.status} used_fallback={result.used_fallback} objective={result.objective_value}")
    if result.used_fallback:
        print(f"[offline] fallback_reason={result.fallback_reason}")
    out = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    write_assignment(result, str(out))


def build_env(cfg, instance_name: str, seed: int, smoke_test: bool) -> EBusDroneEnv:
    instance_data = load_instance(instance_name, seed)
    scenario_data = load_scenario(instance_name, seed)
    assignment_data = load_offline_assignment(instance_name, seed)
    return EBusDroneEnv(config=cfg, instance=instance_data, scenario=scenario_data, assignment=assignment_data, smoke_test=smoke_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate', 'offline', 'train', 'eval', 'benchmark', 'ablation', 'sensitivity', 'all'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--method', default='no_charging')
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base_cfg = load_yaml('configs/default.yaml')
    if 'paths' not in cfg:
        cfg['paths'] = base_cfg.get('paths', {})

    if args.mode in {'generate', 'all'}:
        run_generate(cfg, args.instance, args.seed)
    if args.mode in {'offline', 'all'}:
        run_offline(cfg, args.instance, args.seed)

    if args.mode == 'train':
        env = build_env(cfg, args.instance, args.seed, args.smoke_test)
        train_agent(env, method=args.method, episodes=2 if args.smoke_test else 20, max_steps=10 if args.smoke_test else 100)
    if args.mode == 'eval':
        env = build_env(cfg, args.instance, args.seed, args.smoke_test)
        m = evaluate_policy(env, build_policy(args.method), episodes=1, max_steps=10 if args.smoke_test else 50)
        print(json.dumps(m))
    if args.mode == 'benchmark':
        methods = load_yaml(args.config).get('methods', [])
        for seed in args.seeds:
            env = build_env(cfg, args.instance, seed, args.smoke_test)
            out = Path(cfg['paths']['outputs']) / 'metrics' / f"benchmark_seed_{seed}.json"
            run_benchmark(methods, str(out), env=env, smoke_test=args.smoke_test)
    if args.mode == 'ablation':
        for seed in args.seeds:
            env = build_env(cfg, args.instance, seed, args.smoke_test)
            out = Path(cfg['paths']['outputs']) / 'metrics' / f"ablation_seed_{seed}.json"
            run_ablation(str(out), env=env, smoke_test=args.smoke_test)
    if args.mode == 'sensitivity':
        methods = load_yaml(args.config).get('methods', ['proposed'])
        for seed in args.seeds:
            env = build_env(cfg, args.instance, seed, args.smoke_test)
            out = Path(cfg['paths']['outputs']) / 'metrics' / f"sensitivity_seed_{seed}.json"
            run_sensitivity(methods, str(out), env=env, smoke_test=args.smoke_test)
    if args.mode == 'all':
        run_generate(cfg, args.instance, args.seed)
        run_offline(cfg, args.instance, args.seed)
        env = build_env(cfg, args.instance, args.seed, args.smoke_test)
        train_agent(env, method='proposed', episodes=1 if args.smoke_test else 20, max_steps=10 if args.smoke_test else 100, smoke_test=args.smoke_test)
        run_benchmark(['no_charging','uniform_15','uniform_30','uniform_45','uniform_60','uniform_120','max_feasible','dwell_greedy','battery_threshold','dqn_dr','ddqn_dr','am_ddqn_dr','proposed'], str(Path(cfg['paths']['outputs'])/'metrics'/'all_benchmark.json'), env=env, smoke_test=args.smoke_test)
        run_ablation(str(Path(cfg['paths']['outputs'])/'metrics'/'all_ablation.json'), env=env, smoke_test=args.smoke_test)


if __name__ == '__main__':
    main()
