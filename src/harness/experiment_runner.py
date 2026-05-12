from __future__ import annotations

import json
from pathlib import Path

from src.harness.benchmark_runner import run_benchmark


def run_experiment(name:str, outputs:str, payload:dict)->str:
    p=Path(outputs)/"metrics"/f"{name}.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2), encoding="utf-8")
    return str(p)


def run_methods(methods, outputs, name="benchmark"):
    res = run_benchmark(methods, str(Path(outputs)/"metrics"/f"{name}.json"))
    return res
