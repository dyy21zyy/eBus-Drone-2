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
