from __future__ import annotations
from src.harness.benchmark_runner import run_benchmark

def run_sensitivity(methods, out_path:str, env=None, smoke_test: bool=False):
    return run_benchmark(methods, out_path, env=env, smoke_test=smoke_test)
