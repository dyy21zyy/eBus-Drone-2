from __future__ import annotations


def step_passengers(current: int, arrivals: int, served: int) -> int:
    return max(current + max(arrivals, 0) - max(served, 0), 0)


def simulate_passenger_stop(queue: int, onboard: int, capacity: int, alighting: int, arrivals_during: int) -> dict:
    onboard_after_alight = max(onboard - max(alighting, 0), 0)
    available = max(capacity - onboard_after_alight, 0)
    initial_board = min(max(queue, 0), available)
    onboard_after_initial = onboard_after_alight + initial_board
    queue_after_initial = max(queue - initial_board, 0)

    queue_mid = queue_after_initial + max(arrivals_during, 0)
    available_mid = max(capacity - onboard_after_initial, 0)
    board_mid = min(queue_mid, available_mid)
    onboard_final = onboard_after_initial + board_mid
    queue_final = queue_mid - board_mid

    return {
        "initial_board": initial_board,
        "extra_board": board_mid,
        "total_board": initial_board + board_mid,
        "onboard_final": onboard_final,
        "queue_final": queue_final,
    }
