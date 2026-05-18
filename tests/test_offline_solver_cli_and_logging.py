import json
from pathlib import Path

import pytest

from src.main import main as main_entry
from src.utils.config import load_yaml
from src.main import run_offline


def test_cli_accepts_offline_solver_args(monkeypatch, tmp_path):
    out = tmp_path / 'out'
    argv = ['prog','--mode','generate','--config','configs/debug.yaml','--instance','small','--seed','1','--output-dir',str(out),
            '--offline-solver','scipy','--offline-time-limit','12','--offline-mip-gap','0.2','--offline-log','--offline-solver-disp']
    monkeypatch.setattr('sys.argv', argv)
    main_entry()


def test_scipy_offline_logging_and_metadata(capsys, tmp_path):
    cfg = load_yaml('configs/debug.yaml')
    cfg['paths']['outputs'] = str(tmp_path / 'o')
    cfg['offline']['solver'] = 'scipy'
    run_offline(cfg, 'small', 1)
    out = Path(cfg['paths']['outputs']) / 'assignments' / 'offline_assignment_small_seed_1.json'
    assert out.exists()
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload.get('metadata', {}).get('solver') == 'scipy'
    logs = capsys.readouterr().out
    assert '[offline]' in logs
    assert '[offline][solver]' in logs


def test_gurobi_unavailable_behavior(monkeypatch):
    from src.offline.assignment_data_builder import build_assignment_data
    from src.offline.assignment_solver import solve_assignment, AssignmentInfeasibleError
    from src.main import run_generate
    cfg = load_yaml('configs/debug.yaml')
    run_generate(cfg, 'small', 1)
    inst = json.loads((Path('data/generated/small/instance_seed_1.json')).read_text(encoding='utf-8'))
    data = build_assignment_data(inst)
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == 'gurobipy':
            raise ImportError('mock')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(AssignmentInfeasibleError):
        solve_assignment(data, allow_greedy_fallback=False, solver='gurobi')

def test_gurobi_backend_smoke_if_available(tmp_path):
    gp = pytest.importorskip('gurobipy')
    _ = gp
    cfg = load_yaml('configs/debug.yaml')
    cfg['paths']['outputs'] = str(tmp_path / 'g')
    cfg['offline']['solver'] = 'gurobi'
    cfg['offline']['allow_greedy_fallback'] = False
    run_offline(cfg, 'small', 1)
    out = Path(cfg['paths']['outputs']) / 'assignments' / 'offline_assignment_small_seed_1.json'
    payload = json.loads(out.read_text(encoding='utf-8'))
    meta = payload.get('metadata', {})
    assert meta.get('solver') == 'gurobi'
    assert 'solver_status' in meta
    assert 'runtime_sec' in meta
