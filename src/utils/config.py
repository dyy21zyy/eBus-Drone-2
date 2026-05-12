from pathlib import Path
import json


def _coerce(v: str):
    v = v.strip()
    if v in {"true", "false"}:
        return v == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        if v.startswith('[') and v.endswith(']'):
            return json.loads(v.replace("'", '"'))
        return v


def _simple_yaml_parse(text: str) -> dict:
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        key, _, val = raw.strip().partition(':')
        while indent <= stack[-1][0] and len(stack) > 1:
            stack.pop()
        cur = stack[-1][1]
        if val.strip() == '':
            cur[key] = {}
            stack.append((indent, cur[key]))
        else:
            cur[key] = _coerce(val)
    return root


def load_yaml(path: str) -> dict:
    text = Path(path).read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        return _simple_yaml_parse(text)
