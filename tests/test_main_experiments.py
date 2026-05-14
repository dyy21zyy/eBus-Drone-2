import argparse
import json
from pathlib import Path

import pytest

from src.main import _run_formal_preflight, main
from src.utils.metrics import REQUIRED_PAPER_METRICS


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
    p = out / 'smoke' / 'results' / 'benchmark' / 'small' / 'summary.csv'
    assert p.exists() and p.read_text().strip()


def test_scalability_plan_resolves_instances(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'o'
    _run(['--mode', 'generate', '--experiment', 'scalability', '--config', 'configs/default.yaml', '--smoke', '--output-dir', str(out)], monkeypatch)
    printed = capsys.readouterr().out
    plan = json.loads(printed)
    assert plan['instances'] == ['small', 'medium', 'large']


def test_multiple_seeds_output(tmp_path, monkeypatch, temp_config):
    _run(['--mode', 'generate', '--config', str(temp_config), '--instance', 'small', '--seeds', '1', '2'], monkeypatch)
    base = tmp_path / 'data' / 'generated' / 'small'
    assert (base / 'instance_seed_1.json').exists()
    assert (base / 'instance_seed_2.json').exists()


def test_unknown_method_fails(monkeypatch):
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'eval', '--config', 'configs/default.yaml', '--method', 'bad'])
    with pytest.raises(ValueError, match='Unknown method'):
        main()


def test_all_required_methods_accepted_by_cli(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    methods = ['uniform_15', 'uniform_30', 'uniform_45', 'uniform_60', 'uniform_120', 'dwell_greedy', 'battery_threshold', 'dqn_dr', 'ddqn_dr', 'am_ddqn_dr', 'am_dueling_ddqn_dr']
    _run(['--mode', 'generate', '--config', 'configs/default.yaml', '--instance', 'small', '--seed', '1', '--output-dir', str(out)], monkeypatch)
    _run(['--mode', 'offline', '--config', 'configs/default.yaml', '--instance', 'small', '--seed', '1', '--output-dir', str(out)], monkeypatch)
    _run(['--mode', 'eval', '--config', 'configs/default.yaml', '--instance', 'small', '--method', methods[0], '--seed', '1', '--output-dir', str(out), '--smoke'], monkeypatch)


def test_formal_benchmark_rejects_max_steps_without_override(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    with pytest.raises(ValueError, match='Formal experiments require full horizon'):
        _run(['--mode', 'benchmark', '--config', 'configs/default.yaml', '--instance', 'small', '--methods', 'uniform_30', '--seeds', '1', '--max-steps', '5', '--output-dir', str(out)], monkeypatch)


def test_smoke_pipeline_writes_under_smoke_directory(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    _run(['--mode', 'pipeline', '--config', 'configs/default.yaml', '--instance', 'small', '--seeds', '1', '--methods', 'uniform_30', '--smoke', '--output-dir', str(out)], monkeypatch)
    assert (out / 'smoke' / 'results' / 'benchmark' / 'small' / 'summary.csv').exists()


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


def test_export_tables_reads_nested_benchmark_and_generates_csv_tex(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    p = out / 'results' / 'benchmark' / 'medium'
    p.mkdir(parents=True)
    header = ['method', 'seed', 'full_horizon_completed', 'smoke_mode'] + REQUIRED_PAPER_METRICS
    row1 = ['proposed', '1', 'true', 'false'] + ['10'] * len(REQUIRED_PAPER_METRICS)
    row2 = ['proposed', '2', 'true', 'false'] + ['14'] * len(REQUIRED_PAPER_METRICS)
    (p / 'summary.csv').write_text(",".join(header) + "\n" + ",".join(row1) + "\n" + ",".join(row2) + "\n", encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--output-dir', str(out), '--experiment', 'overall'])
    main()
    csv_out = out / 'tables' / 'benchmark_medium_overall.csv'
    tex_out = out / 'tables' / 'benchmark_medium_overall.tex'
    assert csv_out.exists() and csv_out.read_text(encoding='utf-8').strip()
    txt = tex_out.read_text(encoding='utf-8')
    assert tex_out.exists() and '\\begin{tabular}' in txt and '\\end{tabular}' in txt
    assert '12' in csv_out.read_text(encoding='utf-8')


def test_export_tables_refuses_empty_results(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    p = out / 'results' / 'benchmark' / 'medium'
    p.mkdir(parents=True)
    (p / 'summary.csv').write_text('', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--output-dir', str(out), '--experiment', 'overall'])
    with pytest.raises(ValueError, match='non-empty'):
        main()


def test_export_tables_excludes_smoke_by_default(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    p = out / 'results' / 'benchmark' / 'medium'
    p.mkdir(parents=True)
    header = ['method', 'seed', 'full_horizon_completed', 'smoke_mode'] + REQUIRED_PAPER_METRICS
    smoke = ['proposed', '1', 'true', 'true'] + ['10'] * len(REQUIRED_PAPER_METRICS)
    formal = ['proposed', '2', 'true', 'false'] + ['20'] * len(REQUIRED_PAPER_METRICS)
    (p / 'summary.csv').write_text(",".join(header) + "\n" + ",".join(smoke) + "\n" + ",".join(formal) + "\n", encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--output-dir', str(out), '--experiment', 'overall'])
    main()
    txt = (out / 'tables' / 'benchmark_medium_overall.csv').read_text(encoding='utf-8')
    assert ',1,1,' in txt


def test_validate_pipeline_writes_smoke_output(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    _run(['--mode', 'validate_pipeline', '--config', 'configs/default.yaml', '--output-dir', str(out)], monkeypatch)
    smoke_csv = out / 'smoke' / 'metrics' / 'validate_pipeline.csv'
    assert smoke_csv.exists()
    txt = smoke_csv.read_text(encoding='utf-8')
    assert 'smoke' in txt and 'true' in txt.lower()


def test_formal_benchmark_rejects_empty_method_list(tmp_path):
    cfg = {'paths': {'outputs': str(tmp_path / 'o')}}
    plan = {'smoke': False, 'max_steps': None, 'seeds': [1], 'methods': [], 'instances': ['small']}
    with pytest.raises(ValueError, match='method list must be non-empty'):
        _run_formal_preflight(plan, argparse.Namespace(), cfg)


def test_export_tables_fails_on_missing_metric_not_empty_table(tmp_path, monkeypatch):
    out = tmp_path / 'o'
    p = out / 'results' / 'benchmark' / 'small'
    p.mkdir(parents=True)
    header = ['method', 'seed', 'full_horizon_completed', 'smoke_mode'] + REQUIRED_PAPER_METRICS[:-1]
    row = ['proposed', '1', 'true', 'false'] + ['10'] * (len(REQUIRED_PAPER_METRICS) - 1)
    (p / 'summary.csv').write_text(','.join(header) + '\n' + ','.join(row) + '\n', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['prog', '--mode', 'export_tables', '--config', 'configs/default.yaml', '--output-dir', str(out), '--experiment', 'overall'])
    with pytest.raises(KeyError, match='Missing metrics'):
        main()
    assert not (out / 'tables' / 'benchmark_small_overall.csv').exists()
