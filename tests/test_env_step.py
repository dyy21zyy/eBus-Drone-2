from src.env.ebus_drone_env import EBusDroneEnv


def test_env_step_return_signature_and_repair():
    env = EBusDroneEnv()
    obs, _ = env.reset()
    env.state["available_chargers"] = 0
    nxt, reward, terminated, truncated, info = env.step(8)
    assert obs.shape == nxt.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert info["action_repaired"] is True
    assert info["executed_action_index"] == 0


def test_decision_event_triggering_and_ordinary_stop_ignored_for_decision():
    env = EBusDroneEnv()
    env.reset()
    assert env.current_event is not None
    assert env.current_event.is_decision is True
