import json
from pathlib import Path

import pytest

from src.main import main


def _run(argv, monkeypatch):
    monkeypatch.setattr('sys.argv', ['prog'] + argv)
    main()


def test_old_generate_offline_eval_commands_still_work(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    _run(['--mode', 'generate', '--config', 'configs/default.yaml', '--instance', 'small', '--seed', '1', '--output-dir', str(out)], monkeypatch)
    _run(['--mode', 'offline', '--config', 'configs/default.yaml', '--instance', 'small', '--seed', '1', '--output-dir', str(out)], monkeypatch)
    _run(['--mode', 'eval', '--config', 'configs/default.yaml', '--instance', 'small', '--method', 'uniform_30', '--seed', '1', '--output-dir', str(out), '--smoke'], monkeypatch)
    assert (out / 'assignments' / 'offline_assignment_small_seed_1.json').exists()
    assert (out / 'metrics' / 'eval_uniform_30_small.csv').exists()


def test_pipeline_smoke_creates_nonempty_results(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    _run(['--mode', 'pipeline', '--config', 'configs/default.yaml', '--instance', 'small', '--seeds', '1', '--methods', 'uniform_30', 'battery_threshold', '--smoke', '--output-dir', str(out)], monkeypatch)
    p = out / 'results' / 'benchmark' / 'small' / 'summary.csv'
    assert p.exists() and p.read_text().strip()


def test_scalability_plan_resolves_instances(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'o'
    _run(['--mode', 'generate', '--experiment', 'scalability', '--config', 'configs/default.yaml', '--smoke', '--output-dir', str(out)], monkeypatch)
    printed = capsys.readouterr().out
    plan = json.loads(printed)
    assert plan['instances'] == ['small', 'medium', 'large']


def test_multiple_seeds_output(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    _run(['--mode', 'generate', '--config', 'configs/default.yaml', '--instance', 'small', '--seeds', '1', '2', '--output-dir', str(out)], monkeypatch)
    assert Path('data/generated/small/instance_seed_1.json').exists()
    assert Path('data/generated/small/instance_seed_2.json').exists()


def test_unknown_method_fails(monkeypatch):
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'eval', '--config', 'configs/default.yaml', '--method', 'bad'])
    with pytest.raises(ValueError, match='Unknown method'):
        main()


def test_missing_checkpoint_fails_unless_train_if_missing(tmp_path):
    from src.env.ebus_drone_env import EBusDroneEnv
    from src.harness.benchmark_runner import build_policy

    env = EBusDroneEnv(smoke_test=True)
    with pytest.raises(FileNotFoundError):
        build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=False, seed=1, instance_name='small')
    p = build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=True, smoke_test=True, cfg={'paths': {'outputs': str(tmp_path)}, 'rl': {'episodes': 1, 'max_steps_per_episode': 2}}, seed=1, instance_name='small')
    assert p is not None


def test_export_tables_refuses_missing_metrics(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    p = out / 'results' / 'benchmark'
    p.mkdir(parents=True)
    (p / 'summary.csv').write_text('method,total_reward\nfoo,1\n', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--output-dir', str(out)])
    with pytest.raises(KeyError, match='Missing metrics'):
        main()
