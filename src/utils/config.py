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


def validate_config(cfg: dict) -> None:
    charging = cfg.get("charging", {})
    generation = cfg.get("generation", {})
    power = cfg.get("power", {})
    bus = cfg.get("bus", {})

    action_set = [int(v) for v in charging.get("action_set_seconds", [])]
    if 0 not in action_set:
        raise ValueError("charging.action_set_seconds must include 0 seconds.")
    if not action_set:
        raise ValueError("charging.action_set_seconds must be non-empty.")

    u_max = int(charging.get("max_single_stop_seconds", max(action_set)))
    if u_max != max(action_set):
        raise ValueError(
            f"charging.max_single_stop_seconds ({u_max}) must equal max(charging.action_set_seconds) ({max(action_set)})."
        )

    legacy_h = float(generation.get("horizon_minutes", 480))
    t_bus = float(generation.get("bus_operation_horizon_minutes", legacy_h))
    t_del = float(generation.get("delivery_evaluation_horizon_minutes", legacy_h))
    if t_del < t_bus:
        raise ValueError(
            f"generation.delivery_evaluation_horizon_minutes (T_del={t_del}) must be >= generation.bus_operation_horizon_minutes (T_bus={t_bus})."
        )

    station_capacity = float(power.get("station_capacity_kw", 0.0))
    if station_capacity <= 0.0:
        raise ValueError("power.station_capacity_kw must be > 0.")

    battery_capacity = float(bus.get("battery_capacity_kwh", 0.0))
    safety = float(bus.get("safety_battery_kwh", 0.0))
    if not safety < battery_capacity:
        raise ValueError(
            f"bus.safety_battery_kwh ({safety}) must be < bus.battery_capacity_kwh ({battery_capacity})."
        )

    frac_min = float(bus.get("initial_battery_fraction_min", 0.0))
    frac_max = float(bus.get("initial_battery_fraction_max", 0.0))
    init_min = frac_min * battery_capacity
    init_max = frac_max * battery_capacity
    if init_min < 0.0 or init_max < 0.0 or init_min > battery_capacity or init_max > battery_capacity:
        raise ValueError(
            "bus initial battery range must map within [0, bus.battery_capacity_kwh] via initial_battery_fraction_min/max."
        )


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
