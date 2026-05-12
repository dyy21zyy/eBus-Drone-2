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


def build_assignment_indices(assignment: dict) -> dict:
    decisions = assignment.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Malformed assignment: 'decisions' must be a list")

    by_trip_station: dict[tuple[int, int], list[int]] = {}
    by_customer: dict[int, int] = {}
    station_by_customer: dict[int, int] = {}

    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("Malformed assignment: each decision must be an object")
        if "trip_id" not in item or "station_id" not in item or "customer_id" not in item:
            raise ValueError("Malformed assignment decision: expected trip_id, station_id, customer_id")

        trip_id = int(item["trip_id"])
        station_id = int(item["station_id"])
        customer_id = int(item["customer_id"])

        if customer_id in by_customer:
            raise ValueError(f"Malformed assignment: duplicate customer assignment for {customer_id}")

        by_trip_station.setdefault((trip_id, station_id), []).append(customer_id)
        by_customer[customer_id] = trip_id
        station_by_customer[customer_id] = station_id

    return {
        "by_trip_station": by_trip_station,
        "by_customer": by_customer,
        "station_by_customer": station_by_customer,
    }
