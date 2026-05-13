import pytest

from src.env.reward import compute_reward


def test_total_weighted_cost_and_reward():
    r, comps = compute_reward(
        {"passenger_delay": 2, "parcel_lateness": 3, "energy_cost": 4, "power_overload": 5, "battery_safety": 6, "locker_overflow": 7, "terminal_penalty": 0},
        {"alpha_1": 1, "alpha_2": 2, "alpha_3": 3, "alpha_4": 4, "alpha_5": 5, "alpha_6": 6},
    )
    assert comps["total_cost"] == 2 + 6 + 12 + 20 + 30 + 42
    assert r == -comps["total_cost"]
    assert comps["reward"] == r


def test_missing_data_raises_instead_of_silent_zero():
    with pytest.raises(KeyError):
        compute_reward({"passenger_delay": 1}, {})
