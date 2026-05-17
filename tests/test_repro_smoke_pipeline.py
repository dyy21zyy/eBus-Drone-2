import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.smoke
def test_smoke_generate_offline_eval_cli(temp_config):
    cmds = [
        [sys.executable, '-m', 'src.main', '--mode', 'generate', '--config', str(temp_config), '--instance', 'small', '--seed', '1'],
        [sys.executable, '-m', 'src.main', '--mode', 'offline', '--config', str(temp_config), '--instance', 'small', '--seed', '1'],
        [sys.executable, '-m', 'src.main', '--mode', 'eval', '--config', str(temp_config), '--instance', 'small', '--method', 'uniform_30', '--seed', '1', '--smoke'],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)

    root = Path(temp_config).parent
    assert (root / 'outputs' / 'assignments' / 'offline_assignment_small_seed_1.json').exists()
    assert (root / 'outputs' / 'metrics' / 'eval_uniform_30_small_seed_1.csv').exists()


@pytest.mark.unit
def test_readme_cli_examples_match_main_modes():
    readme = Path('README.md').read_text(encoding='utf-8')
    for mode in ('generate', 'offline', 'train', 'eval', 'benchmark', 'ablation', 'sensitivity', 'pipeline', 'export_tables'):
        assert f'--mode {mode}' in readme
