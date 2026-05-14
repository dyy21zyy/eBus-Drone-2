import pytest

from src.harness.result_validation import ResultValidationError, validate_episode_result
from src.utils.metrics import REQUIRED_PAPER_METRICS


def _base_result():
    r = {k: 1.0 for k in REQUIRED_PAPER_METRICS}
    r.update(
        {
            "episode_end_time": 120.0,
            "operating_horizon_min": 120.0,
            "termination_reason": "horizon_reached",
            "full_horizon_completed": True,
            "truncated_by_max_steps": False,
            "late_delivery_count": 0.0,
            "undelivered_parcel_count": 0.0,
            "terminal_undelivered_penalty": 0.0,
        }
    )
    return r


def test_overrun_fails_validation():
    r = _base_result()
    r["episode_end_time"] = 121.0
    with pytest.raises(ResultValidationError, match="exceeds operating_horizon_min"):
        validate_episode_result(r)


def test_missing_terminal_undelivered_penalty_fails():
    r = _base_result()
    del r["terminal_undelivered_penalty"]
    with pytest.raises(ResultValidationError, match="Missing required metrics"):
        validate_episode_result(r)


def test_smoke_truncated_allowed_but_not_paper_ready():
    r = _base_result()
    r["truncated_by_max_steps"] = True
    r["termination_reason"] = "max_steps_truncated"
    r["full_horizon_completed"] = False
    out = validate_episode_result(r, allow_debug_truncation=True)
    assert out["paper_ready"] is False
    assert out["validation_warnings"]


def test_valid_full_horizon_passes_validation():
    out = validate_episode_result(_base_result())
    assert out["validation_status"] == "success"
    assert out["paper_ready"] is True
