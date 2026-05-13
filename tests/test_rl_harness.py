import pytest
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.trainer import train_agent
from src.harness.benchmark_runner import build_policy, run_benchmark, normalize_method_name
from src.harness.sensitivity_runner import run_sensitivity
from src.harness.evaluator import save_eval_metrics, evaluate_policy
from src.policies.dwell_greedy_policy import DwellGreedyPolicy


def test_training_smoke_writes_checkpoint_and_log(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    train_agent(env, method='proposed', episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}}, seed=1, instance_name='small')
    assert (tmp_path / 'checkpoints' / 'proposed_small_seed_1.pt').exists()
    assert (tmp_path / 'metrics' / 'train_log_proposed_small_seed_1.csv').exists()


def test_missing_checkpoint_error(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    with pytest.raises(FileNotFoundError, match='Missing checkpoint'):
        build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=False, seed=1, instance_name='small')


def test_benchmark_generates_summary_csv(tmp_path):
    rows = run_benchmark(['no_charging'], str(tmp_path / 'benchmark.csv'), env_builder=lambda s: EBusDroneEnv(smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}}, smoke_test=True)
    assert rows and (tmp_path / 'benchmark.csv').exists() and (tmp_path / 'benchmark.json').exists()


def test_method_name_normalization():
    assert normalize_method_name('dwell_based_greedy') == 'dwell_greedy'


def test_evaluation_full_horizon_default_not_50():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics['steps'] != 50


def test_max_steps_explicit_truncates():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=3)
    assert metrics['steps'] <= 3


def test_sensitivity_changes_factor(tmp_path):
    out = tmp_path / 'sens.csv'
    rows = run_sensitivity(['no_charging'], str(out), env_builder=lambda s, c: EBusDroneEnv(config=c, smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}}, factor='passenger_intensity', values=[0.75, 1.25], smoke_test=True)
    assert {r['value'] for r in rows} == {0.75, 1.25}
    assert all('requires_regeneration' in r for r in rows)


def test_eval_metrics_saved_per_seed(tmp_path):
    rows=[{'method':'no_charging','instance':'small','seed':1,'total_reward':0.0},{'method':'no_charging','instance':'small','seed':2,'total_reward':0.0}]
    save_eval_metrics(rows, str(tmp_path/'eval.csv'))
    txt=(tmp_path/'eval.csv').read_text()
    assert 'seed' in txt and '2' in txt


def test_empty_rows_raise_error(tmp_path):
    with pytest.raises(ValueError):
        save_eval_metrics([], str(tmp_path/'empty.csv'))
