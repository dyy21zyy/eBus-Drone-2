from __future__ import annotations
from pathlib import Path
import ast
import json


def _parse_scalar(val: str):
    low = val.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def _simple_yaml_parse(text: str) -> dict:
    data: dict = {}
    stack = [(0, data)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip(' '))
        key, _, val = line.strip().partition(':')
        val = val.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]
        if val == '':
            cur[key] = {}
            stack.append((indent + 2, cur[key]))
        else:
            cur[key] = _parse_scalar(val)
    return data


def load_yaml(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        return _simple_yaml_parse(text)


def _load_json_required(path: Path, kind: str, hint_cmds: list[str]) -> dict:
    if not path.exists():
        hints = "\n".join(hint_cmds)
        raise FileNotFoundError(f"Missing {kind} file: {path}. Generate prerequisites first:\n{hints}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_instance(instance_name: str, seed: int, generated_root: str | Path = "data/generated") -> dict:
    path = Path(generated_root) / instance_name / f"instance_seed_{seed}.json"
    return _load_json_required(path, "instance", [
        f"python -m src.main --mode generate --config configs/default.yaml --instance {instance_name} --seed {seed}",
    ])


def load_scenario(instance_name: str, seed: int, scenario_id: int = 0, generated_root: str | Path = "data/generated") -> dict:
    path = Path(generated_root) / instance_name / f"scenario_{scenario_id}_seed_{seed}.json"
    return _load_json_required(path, "scenario", [
        f"python -m src.main --mode generate --config configs/default.yaml --instance {instance_name} --seed {seed}",
    ])
