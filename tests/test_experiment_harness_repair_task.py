from pathlib import Path

import pytest

from src.harness.sensitivity_runner import _factor_flags
from src.main import main


def _run(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"] + argv)
    main()


def test_ablation_config_has_exact_required_methods():
    import yaml
    methods = yaml.safe_load(Path("configs/experiments/ablation.yaml").read_text(encoding="utf-8"))["methods"]
    assert methods == ["dqn_dr", "ddqn_dr", "am_ddqn_dr", "am_dueling_ddqn_dr"]


def test_sensitivity_parameter_alias_supported(tmp_path, monkeypatch):
    out = tmp_path / "o"
    _run(["--mode", "generate", "--config", "configs/default.yaml", "--instance", "small", "--seed", "1", "--output-dir", str(out)], monkeypatch)
    _run(["--mode", "offline", "--config", "configs/default.yaml", "--instance", "small", "--seed", "1", "--output-dir", str(out)], monkeypatch)
    _run([
        "--mode", "sensitivity", "--config", "configs/default.yaml", "--instance", "small",
        "--seed", "1", "--method", "uniform_30", "--output-dir", str(out), "--parameter",
        "passenger_demand_intensity", "--values", "0.75", "--overwrite"
    ], monkeypatch)
    assert (out / "results" / "sensitivity_summary.csv").exists()


def test_offline_resolve_flags_structural_vs_online_only():
    assert _factor_flags("num_customers")["resolve_offline"] is True
    assert _factor_flags("locker_capacity")["resolve_offline"] is True
    assert _factor_flags("freight_trip_availability")["resolve_offline"] is True
    assert _factor_flags("passenger_intensity")["resolve_offline"] is False
    assert _factor_flags("chargers_per_station")["resolve_offline"] is False
    assert _factor_flags("charging_power")["resolve_offline"] is False
    assert _factor_flags("station_power_capacity")["resolve_offline"] is False


def test_named_summary_writer_rejects_empty(tmp_path):
    from src.main import _write_named_summary
    src = tmp_path / "empty.csv"
    src.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty summaries"):
        _write_named_summary(src, tmp_path / "x.csv")
