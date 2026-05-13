import numpy as np

from src.env.passenger_process import simulate_passenger_service_at_stop


def test_passenger_capacity_respected_and_chi_from_queue():
    rng = np.random.default_rng(7)
    out = simulate_passenger_service_at_stop(queue=5, onboard=78, capacity=80, alighting_probability=0.0, rate_per_min=0.0, rho_al_min_per_pax=0.02, rho_bo_min_per_pax=0.05, parcel_unloading_time_min=0.0, charging_duration_min=0.0, rng=rng)
    assert out["onboard_final"] <= 80
    assert out["chi"] is True


def test_dwell_time_can_extend_due_to_arrivals_in_excess_interval():
    rng = np.random.default_rng(0)
    out = simulate_passenger_service_at_stop(queue=0, onboard=0, capacity=80, alighting_probability=0.0, rate_per_min=4.0, rho_al_min_per_pax=0.01, rho_bo_min_per_pax=0.1, parcel_unloading_time_min=2.0, charging_duration_min=0.0, rng=rng)
    assert out["realized_dwell_min"] >= 2.0
    assert out["board_during_excess"] >= 0


def test_reproducible_with_same_seed():
    a = simulate_passenger_service_at_stop(queue=1, onboard=10, capacity=80, alighting_probability=0.2, rate_per_min=0.3, rho_al_min_per_pax=0.02, rho_bo_min_per_pax=0.05, parcel_unloading_time_min=0.0, charging_duration_min=1.0, rng=np.random.default_rng(42))
    b = simulate_passenger_service_at_stop(queue=1, onboard=10, capacity=80, alighting_probability=0.2, rate_per_min=0.3, rho_al_min_per_pax=0.02, rho_bo_min_per_pax=0.05, parcel_unloading_time_min=0.0, charging_duration_min=1.0, rng=np.random.default_rng(42))
    assert a == b
