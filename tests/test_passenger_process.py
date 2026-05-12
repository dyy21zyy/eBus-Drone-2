from src.env.passenger_process import simulate_passenger_stop


def test_passenger_stop_processes_queue_and_capacity():
    out = simulate_passenger_stop(queue=5, onboard=30, capacity=32, alighting=2, arrivals_during=3)
    assert out["onboard_final"] <= 32
    assert out["queue_final"] >= 0
    assert out["initial_board"] >= 0
