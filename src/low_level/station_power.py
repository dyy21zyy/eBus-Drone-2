from __future__ import annotations


def compute_station_power(p_e: float, p_l: float, p_d: float, p_capacity: float) -> dict:
    p_tot = float(p_e) + float(p_d) + float(p_l)
    overload = max(0.0, p_tot - float(p_capacity))
    return {"P_E": float(p_e), "P_L": float(p_l), "P_D": float(p_d), "P_tot": p_tot, "overload": overload}
