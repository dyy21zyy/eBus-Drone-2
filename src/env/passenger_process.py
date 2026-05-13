from __future__ import annotations

from typing import Any


def step_passengers(current: int, arrivals: int, served: int) -> int:
    return max(current + max(arrivals, 0) - max(served, 0), 0)


def sample_poisson_arrivals(rate_per_min: float, interval_min: float, rng: Any) -> int:
    lam = max(rate_per_min, 0.0) * max(interval_min, 0.0)
    return int(rng.poisson(lam))


def sample_alighting(onboard_passengers: int, alighting_probability: float, rng: Any) -> int:
    n = max(int(onboard_passengers), 0)
    p = min(max(float(alighting_probability), 0.0), 1.0)
    return int(rng.binomial(n, p))


def board_from_queue(queue: int, remaining_capacity: int) -> tuple[int, int]:
    q = max(int(queue), 0)
    cap = max(int(remaining_capacity), 0)
    boarded = min(q, cap)
    return boarded, q - boarded


def simulate_arrivals_during_dwell(queue: int, onboard_after_initial: int, capacity: int, rate_per_min: float, interval_min: float, rng: Any) -> dict:
    arrivals = sample_poisson_arrivals(rate_per_min, interval_min, rng)
    queue_with_arrivals = max(int(queue), 0) + arrivals
    remaining_capacity = max(int(capacity) - max(int(onboard_after_initial), 0), 0)
    boarded, queue_after = board_from_queue(queue_with_arrivals, remaining_capacity)
    return {
        "arrivals": arrivals,
        "boarded": boarded,
        "queue_after": queue_after,
    }


def sample_initial_passenger_event(*, queue: int, onboard: int, capacity: int, alighting_probability: float, rho_al_min_per_pax: float, rho_bo_min_per_pax: float, rng: Any) -> dict:
    n_al = sample_alighting(onboard, alighting_probability, rng)
    onboard_after_alight = max(int(onboard) - n_al, 0)
    remaining_capacity = max(int(capacity) - onboard_after_alight, 0)
    n_bo_initial, queue_after_initial = board_from_queue(queue, remaining_capacity)
    onboard_after_initial = onboard_after_alight + n_bo_initial
    passenger_dwell_min = max(max(float(rho_al_min_per_pax), 0.0) * n_al, max(float(rho_bo_min_per_pax), 0.0) * n_bo_initial)
    return {
        "queue_at_arrival": max(int(queue), 0),
        "onboard_before_alight": max(int(onboard), 0),
        "alighting": n_al,
        "initial_board": n_bo_initial,
        "queue_after_initial": queue_after_initial,
        "onboard_after_initial": onboard_after_initial,
        "remaining_capacity_after_initial": max(int(capacity) - onboard_after_initial, 0),
        "passenger_dwell_min": passenger_dwell_min,
        "chi": (n_al + n_bo_initial) > 0,
    }


def finalize_passenger_service(*, initial_event: dict, capacity: int, rate_per_min: float, rho_bo_min_per_pax: float, parcel_unloading_time_min: float, charging_duration_min: float, rng: Any) -> dict:
    passenger_dwell = float(initial_event.get("passenger_dwell_min", 0.0))
    normal = simulate_arrivals_during_dwell(initial_event["queue_after_initial"], initial_event["onboard_after_initial"], capacity, rate_per_min, passenger_dwell, rng)
    onboard_after_normal = int(initial_event["onboard_after_initial"]) + normal["boarded"]

    normal_boarding_extension = max(float(rho_bo_min_per_pax), 0.0) * normal["boarded"]
    passenger_dwell += normal_boarding_extension

    baseline_non_pax = max(max(float(parcel_unloading_time_min), 0.0), max(float(charging_duration_min), 0.0))
    excess_interval = max(0.0, baseline_non_pax - passenger_dwell)
    extra = {"arrivals": 0, "boarded": 0, "queue_after": normal["queue_after"]}
    extra_extension = 0.0
    onboard_final = onboard_after_normal
    if excess_interval > 0.0:
        extra = simulate_arrivals_during_dwell(normal["queue_after"], onboard_after_normal, capacity, rate_per_min, excess_interval, rng)
        onboard_final = onboard_after_normal + extra["boarded"]
        extra_extension = max(float(rho_bo_min_per_pax), 0.0) * extra["boarded"]

    realized_dwell = max(passenger_dwell, baseline_non_pax) + extra_extension
    return {
        "alighting": int(initial_event["alighting"]),
        "initial_board": int(initial_event["initial_board"]),
        "board_during_normal": normal["boarded"],
        "board_during_excess": extra["boarded"],
        "total_board": int(initial_event["initial_board"]) + normal["boarded"] + extra["boarded"],
        "onboard_final": onboard_final,
        "queue_final": extra["queue_after"],
        "passenger_dwell_min": passenger_dwell,
        "realized_dwell_min": realized_dwell,
        "chi": bool(initial_event.get("chi", False)),
    }


def simulate_passenger_service_at_stop(*, queue: int, onboard: int, capacity: int, alighting_probability: float, rate_per_min: float, rho_al_min_per_pax: float, rho_bo_min_per_pax: float, parcel_unloading_time_min: float, charging_duration_min: float, rng: Any) -> dict:
    initial = sample_initial_passenger_event(queue=queue, onboard=onboard, capacity=capacity, alighting_probability=alighting_probability, rho_al_min_per_pax=rho_al_min_per_pax, rho_bo_min_per_pax=rho_bo_min_per_pax, rng=rng)
    return finalize_passenger_service(initial_event=initial, capacity=capacity, rate_per_min=rate_per_min, rho_bo_min_per_pax=rho_bo_min_per_pax, parcel_unloading_time_min=parcel_unloading_time_min, charging_duration_min=charging_duration_min, rng=rng)
