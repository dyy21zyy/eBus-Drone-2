import numpy as np

from src.env.action_space import feasible_action_mask, repair_action


def test_no_charger_only_zero_feasible():
    mask = feasible_action_mask(available_chargers=0, battery=50, battery_max=150, charge_rate=1)
    assert mask.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_near_full_battery_masks_long_actions():
    mask = feasible_action_mask(available_chargers=1, battery=148, battery_max=150, charge_rate=0.1)
    assert mask[0] == 1
    assert mask[-1] == 0


def test_station_power_not_in_mask():
    mask1 = feasible_action_mask(1, 100, 150, 0.5)
    mask2 = feasible_action_mask(1, 100, 150, 0.5)
    assert np.array_equal(mask1, mask2)


def test_infeasible_repaired_to_feasible():
    mask = feasible_action_mask(1, 149, 150, 0.01)
    repaired = repair_action(8, mask)
    assert mask[repaired] == 1
    assert repaired <= 8
