from pathlib import Path
import json


def write_instance_and_scenarios(instance: dict, scenarios: list[dict], out_dir: str) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    ip = p / "instance.json"
    ip.write_text(json.dumps(instance, indent=2), encoding='utf-8')
    scenario_files = []
    for i, sc in enumerate(scenarios, start=1):
        sp = p / f"scenario_{i}.json"
        sp.write_text(json.dumps(sc, indent=2), encoding='utf-8')
        scenario_files.append(str(sp))
    return {"instance": str(ip), "scenarios": scenario_files}
