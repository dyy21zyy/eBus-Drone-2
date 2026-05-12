from __future__ import annotations
from pathlib import Path
import ast


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
