from __future__ import annotations


def charged_energy_kwh(power_kw: float, duration_sec: float, eta: float) -> float:
    """Compute delivered charging energy in kWh from kW and seconds."""
    return max(0.0, eta) * max(0.0, power_kw) * max(0.0, duration_sec) / 3600.0


def apply_charge(
    battery_kwh: float,
    duration_sec: float,
    power_kw: float,
    eta: float,
    battery_capacity_kwh: float,
) -> float:
    return min(battery_capacity_kwh, battery_kwh + charged_energy_kwh(power_kw, duration_sec, eta))


def apply_travel_consumption(
    battery_kwh: float,
    distance_km: float,
    energy_consumption_kwh_per_km: float,
) -> float:
    return battery_kwh - max(0.0, distance_km) * max(0.0, energy_consumption_kwh_per_km)
