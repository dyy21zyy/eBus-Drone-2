from __future__ import annotations


def check_termination(state: dict, has_future_decision: bool) -> tuple[bool, str | None]:
    if state["time"] >= state["horizon"]:
        return True, "horizon_reached"
    if state["battery"] <= 0.0:
        return True, "battery_depleted"
    if state.get("infeasible", False):
        return True, "severe_infeasibility"
    if not has_future_decision:
        return True, "no_future_decision"
    return False, None
