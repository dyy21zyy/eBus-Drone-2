from __future__ import annotations


def check_termination(state: dict, has_future_decision: bool) -> tuple[bool, str | None]:
    if state.get("time", 0.0) >= state.get("horizon", float("inf")):
        return True, "horizon_reached"
    if not has_future_decision:
        return True, "no_future_decision"
    return False, None


def apply_terminal_penalty_once(state: dict, undelivered: list[dict], t_end: float, eta_l_term: float, eta_u_term: float) -> float:
    if state.get("terminal_penalty_applied", False):
        return 0.0
    state["terminal_penalty_applied"] = True
    penalty = 0.0
    for p in undelivered:
        if p.get("status") == "delivered":
            continue
        penalty += eta_l_term * max(0.0, float(t_end) - float(p.get("deadline_min", p.get("deadline", t_end)))) + eta_u_term
    return penalty
