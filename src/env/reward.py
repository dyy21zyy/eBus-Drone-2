from __future__ import annotations


def _required_float(mapping: dict, key: str) -> float:
    if key not in mapping:
        raise KeyError(f"Missing reward component: {key}")
    return float(mapping[key])


def compute_reward(components: dict, alphas: dict) -> tuple[float, dict]:
    """Paper reward: R = -(a1*D_P + a2*D_L + a3*D_E + a4*D_Pwr + a5*D_B + a6*D_K)."""
    d_p = _required_float(components, "passenger_delay")
    d_l = _required_float(components, "parcel_lateness")
    d_e = _required_float(components, "energy_cost")
    d_pwr = _required_float(components, "power_overload")
    d_b = _required_float(components, "battery_safety")
    d_k = _required_float(components, "locker_overflow")

    total_cost = (
        float(alphas.get("alpha_1", 1.0)) * d_p
        + float(alphas.get("alpha_2", 1.0)) * d_l
        + float(alphas.get("alpha_3", 1.0)) * d_e
        + float(alphas.get("alpha_4", 1.0)) * d_pwr
        + float(alphas.get("alpha_5", 1.0)) * d_b
        + float(alphas.get("alpha_6", 1.0)) * d_k
    )
    reward = -float(total_cost)
    out = dict(components)
    out["total_cost"] = float(total_cost)
    out["reward"] = reward
    return reward, out
