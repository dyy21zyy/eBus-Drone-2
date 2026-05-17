from __future__ import annotations
from src.harness.benchmark_runner import run_benchmark
from src.utils.config import load_yaml

def run_ablation(out_csv:str, env_builder, instance_name:str, test_seeds:list[int], cfg:dict, smoke_test: bool=False, train_if_missing: bool=False):
    methods=load_yaml('configs/experiments/ablation.yaml').get('methods', ['dqn_dr','ddqn_dr','am_ddqn_dr','am_dueling_ddqn_dr'])
    return run_benchmark(methods, out_csv, env_builder=env_builder, instance_name=instance_name, test_seeds=test_seeds, cfg=cfg, smoke_test=smoke_test, train_if_missing=train_if_missing)
