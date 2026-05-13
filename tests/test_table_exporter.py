from __future__ import annotations

import csv
from pathlib import Path
import pytest

from src.harness.table_exporter import export_tables
from src.main import main


def _write_summary(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _base_row(method: str, seed: int, total_cost: float):
    return {
        "method": method, "seed": str(seed), "total_cost": total_cost, "total_reward": 1.0,
        "onboard_passenger_delay": 2.0, "average_excess_dwell_time": 0.0, "total_bus_operating_delay": 0.0,
        "parcel_lateness": 3.0, "late_delivery_count": 1.0, "undelivered_parcel_count": 0.0,
        "average_locker_holding_time": 0.0, "terminal_undelivered_penalty": 0.0, "minimum_bus_battery": 12.0,
        "battery_safety_violation_count": 0.0, "total_energy_consumption": 5.0,
        "station_power_overload_amount": 0.0, "station_power_overload_duration": 0.0,
        "locker_overflow_amount": 0.0, "locker_overflow_duration": 0.0, "charger_utilization": 0.0,
        "drone_battery_stockout_count": 0.0, "runtime_sec": 0.2, "evaluation_time_sec": 0.1,
    }


def test_aggregation_mean_std_and_order(tmp_path):
    rows = [_base_row("m2", 1, 10.0), _base_row("m1", 1, 20.0), _base_row("m1", 2, 24.0)]
    p = tmp_path / "outputs" / "results" / "overall" / "small" / "summary.csv"
    _write_summary(p, rows)
    res = export_tables(tmp_path / "outputs", "overall", "small", ["m1", "m2"])
    agg = Path(res.aggregated_csv).read_text(encoding="utf-8")
    assert "m1" in agg and "22.0000 ± 2.0000" in agg
    first_data = agg.splitlines()[1]
    assert first_data.startswith("m1,")


def test_missing_metric_raises(tmp_path):
    p = tmp_path / "outputs" / "results" / "overall" / "small" / "summary.csv"
    _write_summary(p, [{"method": "m1", "seed": "1", "total_cost": 1.0}])
    with pytest.raises(KeyError):
        export_tables(tmp_path / "outputs", "overall", "small", ["m1"])


def test_empty_input_raises(tmp_path):
    p = tmp_path / "outputs" / "results" / "overall" / "small" / "summary.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("method,seed,total_cost\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty input"):
        export_tables(tmp_path / "outputs", "overall", "small", ["m1"], metrics_subset=["total_cost"])


def test_latex_contains_aggregated_values(tmp_path):
    p = tmp_path / "outputs" / "results" / "overall" / "small" / "summary.csv"
    _write_summary(p, [_base_row("m1", 1, 5.0), _base_row("m1", 2, 7.0)])
    res = export_tables(tmp_path / "outputs", "overall", "small", ["m1"], metrics_subset=["total_cost"])
    tex = Path(res.tex_path).read_text(encoding="utf-8")
    assert "6.0000 ± 1.0000" in tex


def test_export_tables_cli_smoke(monkeypatch, tmp_path):
    out = tmp_path / "o"
    p = out / "results" / "overall" / "small" / "summary.csv"
    _write_summary(p, [_base_row("m1", 1, 9.0)])
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--experiment', 'overall', '--instance', 'small', '--output-dir', str(out), '--metrics', 'total_cost'])
    main()
    assert (out / 'tables' / 'overall_small.csv').exists()
