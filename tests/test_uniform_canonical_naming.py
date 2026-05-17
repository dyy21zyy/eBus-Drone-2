import csv
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import run_benchmark


def test_benchmark_uniform_uses_canonical_method_and_duration_field(tmp_path):
    out_csv = tmp_path / "benchmark.csv"
    rows = run_benchmark(["uniform_45"], str(out_csv), env_builder=lambda s: EBusDroneEnv(smoke_test=True), instance_name="small", test_seeds=[1], cfg={"paths": {"outputs": str(tmp_path)}, "uniform_duration_sec": 45}, smoke_test=True)
    assert rows[0]["method"] == "uniform"
    assert rows[0]["uniform_duration_sec"] == 45
    with out_csv.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["method"] == "uniform"
    assert "uniform_duration_sec" in row
    assert not (tmp_path / "metrics" / "eval_uniform_45_small_seed_1.csv").exists()
