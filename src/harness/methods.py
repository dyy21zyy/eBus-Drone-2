from __future__ import annotations

CANONICAL_METHODS = {
    "uniform",
    "dwell_greedy",
    "battery_threshold",
    "dqn_dr",
    "ddqn_dr",
    "am_ddqn_dr",
    "am_dueling_ddqn_dr",
}

METHOD_ALIASES = {
    "proposed": "am_dueling_ddqn_dr",
    "dwell_based_greedy": "dwell_greedy",
}


def normalize_method_name(method: str) -> str:
    if method is None:
        raise ValueError("Method cannot be None")
    m = str(method).strip().lower()
    if m == "uniform":
        return "uniform_45"
    return METHOD_ALIASES.get(m, m)


def is_canonical_method(method: str) -> bool:
    return normalize_method_name(method) in CANONICAL_METHODS
