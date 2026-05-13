from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from src.data_generation.instance_writer import write_instance
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.ablation_runner import run_ablation
from src.harness.benchmark_runner import build_policy, run_benchmark
from src.harness.evaluator import evaluate_policy, save_eval_metrics
from src.harness.sensitivity_runner import FACTOR_PATHS, run_sensitivity
from src.harness.trainer import train_agent
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import load_offline_assignment, write_assignment
from src.offline.assignment_solver import solve_assignment
from src.offline.assignment_validator import summarize_assignment_flows
from src.utils.config import load_instance, load_scenario, load_yaml
from src.utils.metrics import REQUIRED_PAPER_METRICS
from src.utils.random_seed import set_seed

VALID_METHODS = {
    "uniform_30", "dwell_based_greedy", "dwell_greedy", "battery_threshold", "dqn_dr", "ddqn_dr", "am_ddqn_dr", "proposed", "no_charging", "max_feasible",
}


def _require_exists(path: Path, msg: str):
    if not path.exists():
        raise FileNotFoundError(msg)


def run_generate(cfg, instance_name: str, seed: int):
    cfg_file = Path(f"configs/instances/{instance_name}.yaml")
    _require_exists(cfg_file, f"Missing instance file: {cfg_file}")
    instance_cfg = load_yaml(str(cfg_file))
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
    assign_path = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    _require_exists(assign_path, f"Missing offline assignment: {assign_path}. Run --mode offline first.")
    return EBusDroneEnv(config=cfg, instance=load_instance(instance_name, seed), scenario=load_scenario(instance_name, seed), assignment=load_offline_assignment(instance_name, seed), smoke_test=smoke_test)


def _save_run_metadata(run_dir: Path, cfg: dict, args: argparse.Namespace, plan: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "cli_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def _validate_args(args):
    if args.method and args.methods and args.method != "no_charging":
        raise ValueError("Use either --method or --methods, not both.")
    if args.mode != "sensitivity" and args.sensitivity:
        raise ValueError("--sensitivity is only valid for --mode sensitivity.")
    if args.mode == "sensitivity" and not args.sensitivity:
        raise ValueError("Sensitivity mode requires --sensitivity.")
    if args.sensitivity and args.sensitivity not in FACTOR_PATHS:
        raise ValueError(f"Unknown sensitivity variable: {args.sensitivity}")
    for m in args.methods or []:
        if m not in VALID_METHODS:
            raise ValueError(f"Unknown method: {m}")


def _resolve_plan(args):
    instances = args.instances or [args.instance]
    seeds = args.seeds or ([args.seed] if args.seed is not None else [1])
    methods = args.methods or ([args.method] if args.method else ["no_charging"])
    if args.experiment == "scalability":
        instances = ["small", "medium", "large"]
    return {"mode": args.mode, "experiment": args.experiment, "instances": instances, "seeds": seeds, "methods": methods, "max_steps": args.max_steps, "smoke": bool(args.smoke)}


def _run_eval(cfg, instance, seed, method, args, rows):
    env = build_env(cfg, instance, seed, args.smoke)
    try:
        pol = build_policy(method, env=env, out_root=cfg['paths']['outputs'], checkpoint=args.checkpoint, train_if_missing=args.train_if_missing, smoke_test=args.smoke, cfg=cfg, seed=seed, instance_name=instance)
    except TypeError:
        pol = build_policy(method)
    m = evaluate_policy(env, pol, episodes=1, max_steps=args.max_steps if args.max_steps is not None else (10 if args.smoke else None))
    m.update({'method': method, 'instance': instance, 'seed': seed})
    rows.append(m)


def _export_tables(output_dir: Path, experiment_name: str):
    src_csv = output_dir / "results" / experiment_name / "summary.csv"
    _require_exists(src_csv, f"Missing results CSV for export_tables: {src_csv}")
    rows = list(csv.DictReader(src_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError("Empty results: export_tables requires non-empty metrics rows.")
    missing = [k for k in REQUIRED_PAPER_METRICS if k not in rows[0]]
    if missing:
        raise KeyError(f"Missing metrics for export_tables: {missing}")
    out_csv = output_dir / "tables" / f"{experiment_name}_paper.csv"
    out_tex = output_dir / "tables" / f"{experiment_name}_paper.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    tex_lines = ["\\begin{tabular}{lrr}", "method & total_reward & total_cost \\\\", "\\hline"]
    for r in rows:
        tex_lines.append(f"{r['method']} & {r.get('total_reward','')} & {r.get('total_cost','')} \\\\")
    tex_lines.append("\\end{tabular}")
    out_tex.write_text("\n".join(tex_lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate', 'offline', 'train', 'eval', 'benchmark', 'ablation', 'sensitivity', 'pipeline', 'export_tables'])
    ap.add_argument('--experiment', choices=['overall', 'ablation', 'scalability', 'sensitivity'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--instances', nargs='+')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--method', default='no_charging')
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--episodes', type=int)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--smoke-test', action='store_true')
    ap.add_argument('--checkpoint'); ap.add_argument('--train-if-missing', action='store_true')
    ap.add_argument('--factor'); ap.add_argument('--values', nargs='+', type=float)
    ap.add_argument('--sensitivity')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--output-dir')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.smoke_test:
        args.smoke = True
    _validate_args(args)
    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg['paths']['outputs'] = args.output_dir
    plan = _resolve_plan(args)
    print(json.dumps(plan, indent=2))
    run_dir = Path(cfg['paths']['outputs']) / 'runs' / f"{plan['mode']}_{(args.experiment or 'single')}"
    _save_run_metadata(run_dir, cfg, args, plan)

    if args.mode == 'generate':
        for i in plan['instances']:
            for s in plan['seeds']:
                run_generate(cfg, i, s)
        return
    if args.mode == 'offline':
        for i in plan['instances']:
            for s in plan['seeds']:
                run_offline(cfg, i, s)
        return
    if args.mode == 'train':
        for i in plan['instances']:
            for seed in plan['seeds']:
                set_seed(seed, deterministic=bool(cfg.get('rl', {}).get('deterministic', False)))
                env = build_env(cfg, i, seed, args.smoke)
                train_agent(env, method=plan['methods'][0], episodes=args.episodes or (2 if args.smoke else 20), max_steps=args.max_steps if args.max_steps is not None else (10 if args.smoke else 100), smoke_test=args.smoke, out_root=cfg['paths']['outputs'], cfg=cfg, seed=seed, instance_name=i)
        return
    if args.mode == 'eval':
        rows = []
        for i in plan['instances']:
            for seed in plan['seeds']:
                _run_eval(cfg, i, seed, plan['methods'][0], args, rows)
        out = Path(cfg['paths']['outputs']) / 'metrics' / f"eval_{plan['methods'][0]}_{plan['instances'][0]}.csv"
        save_eval_metrics(rows, str(out)); print(json.dumps(rows)); return
    if args.mode in ('benchmark', 'ablation', 'sensitivity', 'pipeline'):
        if args.mode == 'pipeline':
            for i in plan['instances']:
                for s in plan['seeds']:
                    run_generate(cfg, i, s); run_offline(cfg, i, s)
        if args.mode in ('benchmark', 'pipeline'):
            for i in plan['instances']:
                out = Path(cfg['paths']['outputs']) / 'results' / (args.experiment or 'benchmark') / i / 'summary.csv'
                if out.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists and overwrite is disabled: {out}")
                run_benchmark(plan['methods'], str(out), env_builder=lambda sd, _i=i: build_env(cfg, _i, sd, args.smoke), instance_name=i, test_seeds=plan['seeds'], cfg=cfg, smoke_test=args.smoke, train_if_missing=args.train_if_missing)
        if args.mode == 'ablation':
            for i in plan['instances']:
                out = Path(cfg['paths']['outputs']) / 'results' / 'ablation' / i / 'summary.csv'
                run_ablation(str(out), env_builder=lambda sd, _i=i: build_env(cfg, _i, sd, args.smoke), instance_name=i, test_seeds=plan['seeds'], cfg=cfg, smoke_test=args.smoke, train_if_missing=args.train_if_missing)
        if args.mode == 'sensitivity':
            vals = args.values or [1.0]
            for i in plan['instances']:
                out = Path(cfg['paths']['outputs']) / 'results' / 'sensitivity' / i / f'{args.sensitivity}.csv'
                run_sensitivity(plan['methods'], str(out), env_builder=lambda sd, c, _i=i: build_env(c, _i, sd, args.smoke), instance_name=i, test_seeds=plan['seeds'], cfg=cfg, factor=args.sensitivity, values=vals, smoke_test=args.smoke, train_if_missing=args.train_if_missing)
        return
    if args.mode == 'export_tables':
        _export_tables(Path(cfg['paths']['outputs']), args.experiment or 'benchmark')
        return


if __name__ == '__main__':
    main()
