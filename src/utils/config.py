from __future__ import annotations
from pathlib import Path


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
            if val.isdigit():
                cur[key] = int(val)
            else:
                cur[key] = val
    return data


def load_yaml(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        return _simple_yaml_parse(text)
