from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_generation.instance_writer import write_instance
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import build_policy, run_benchmark
from src.harness.evaluator import evaluate_policy
from src.harness.trainer import train_agent
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import write_assignment
from src.offline.assignment_solver import solve_assignment
from src.utils.config import load_yaml


def run_generate(cfg, instance_name: str, seed: int):
    instance_cfg = load_yaml(f"configs/instances/{instance_name}.yaml")
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    for idx in range(int(cfg["generation"].get("num_scenarios", 1))):
        write_instance(generate_scenario(cfg, instance, seed, idx), out_root, f"scenario_{idx}_seed_{seed}")


def run_offline(cfg, instance_name: str, seed: int):
    instance_path = Path(cfg["paths"]["data_generated"]) / instance_name / f"instance_seed_{seed}.json"
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    result = solve_assignment(build_assignment_data(instance))
    out = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    write_assignment(result, str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate','offline','train','eval','benchmark','ablation','all'])
    ap.add_argument('--config', required=True); ap.add_argument('--instance', default='small')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--seeds', default='1')
    ap.add_argument('--method', default='no_charging'); ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args(); cfg = load_yaml(args.config)
    base_cfg = load_yaml('configs/default.yaml')
    if args.mode in {'generate','all'}: run_generate(cfg,args.instance,args.seed)
    if args.mode in {'offline','all'}: run_offline(cfg,args.instance,args.seed)
    if args.mode == 'train':
        agent = train_agent(EBusDroneEnv(), method=args.method, episodes=2 if args.smoke_test else 20, max_steps=10 if args.smoke_test else 100)
        out = Path((cfg if 'paths' in cfg else base_cfg)['paths']['outputs'])/'checkpoints'; out.mkdir(parents=True,exist_ok=True)
        import torch; torch.save(agent.online.state_dict(), out/f"{args.method}_seed_{args.seed}.pt")
    if args.mode == 'eval':
        m = evaluate_policy(EBusDroneEnv(), build_policy(args.method), episodes=1, max_steps=10 if args.smoke_test else 50)
        print(json.dumps(m))
    if args.mode in {'benchmark','ablation'}:
        ecfg = load_yaml(args.config)
        methods = ecfg.get('methods', [])
        run_benchmark(methods, str(Path((cfg if 'paths' in cfg else base_cfg)['paths']['outputs'])/'metrics'/f"{args.mode}.json"))
    if args.mode == 'all':
        methods = ['no_charging','max_feasible','dqn_dr','ddqn_dr','am_ddqn_dr','proposed']
        run_benchmark(methods, str(Path((cfg if 'paths' in cfg else base_cfg)['paths']['outputs'])/'metrics'/'all.json'))


if __name__ == '__main__':
    main()
