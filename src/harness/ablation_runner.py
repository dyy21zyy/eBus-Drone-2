from __future__ import annotations
from src.harness.benchmark_runner import run_benchmark

def run_ablation(out_path:str, env=None, smoke_test: bool=False):
    return run_benchmark(["dqn_dr","ddqn_dr","am_ddqn_dr","proposed"], out_path, env=env, smoke_test=smoke_test)
