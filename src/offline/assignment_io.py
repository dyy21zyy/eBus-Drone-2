from __future__ import annotations

import json
from pathlib import Path

from src.offline.assignment_result import AssignmentResult


def write_assignment(result: AssignmentResult, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def read_assignment(path: str) -> AssignmentResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AssignmentResult.from_dict(payload)


def load_offline_assignment(instance_name: str, seed: int) -> dict:
    path = Path("outputs/assignments") / f"offline_assignment_{instance_name}_seed_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            "Missing offline assignment file: "
            f"{path}. Run prerequisite commands first:\n"
            f"python -m src.main --mode generate --config configs/default.yaml --instance {instance_name} --seed {seed}\n"
            f"python -m src.main --mode offline --config configs/default.yaml --instance {instance_name} --seed {seed}"
        )
    return json.loads(path.read_text(encoding="utf-8"))
