import pytest
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.trainer import train_agent
from src.harness.benchmark_runner import build_policy, run_benchmark, normalize_method_name, uniform_seconds_from_method
from src.harness.sensitivity_runner import run_sensitivity
from src.harness.evaluator import save_eval_metrics, evaluate_policy
from src.policies.dwell_greedy_policy import DwellGreedyPolicy
import csv


class _LongEpisodeEnv:
    def __init__(self, done_after: int = 5):
        self.done_after = done_after
        self.horizon_sec = 600.0
        self.state = {"time": 0.0}
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        self.state = {"time": 0.0}
        return [0.0, 0.0], {}

    def get_action_mask(self):
        return [1, 1]

    def step(self, _action):
        self._steps += 1
        self.state["time"] = float(self._steps * 60.0)
        done = self._steps >= self.done_after
        reason = "horizon_reached" if done else None
        return [0.0, 0.0], 0.0, done, False, {"termination_reason": reason, "reward_components": {}}


def test_training_smoke_writes_checkpoint_and_log(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(env, method='proposed', episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}}, seed=1, instance_name='small')
    assert (tmp_path / 'checkpoints' / 'proposed_small_seed_1.pt').exists()
    assert (tmp_path / 'metrics' / 'train_log_proposed_small_seed_1.csv').exists()


def test_formal_training_ignores_max_steps_and_runs_to_done(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(
        env,
        method='proposed',
        episodes=1,
        max_steps=2,
        smoke_test=False,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_proposed_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert int(row['episode_steps']) > 2
    assert row['truncated_by_max_steps'].lower() == 'false'


def test_smoke_training_allows_max_step_truncation_and_labels_non_paper_ready(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(
        env,
        method='proposed',
        episodes=1,
        max_steps=2,
        smoke_test=True,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_proposed_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert row['truncated_by_max_steps'].lower() == 'true'
    assert row['termination_reason'] == 'max_steps_truncated'
    assert row['paper_ready_episode'].lower() == 'false'


def test_missing_checkpoint_error(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    with pytest.raises(FileNotFoundError, match='Missing checkpoint'):
        build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=False, seed=1, instance_name='small')


def test_benchmark_generates_summary_csv(tmp_path):
    rows = run_benchmark(['no_charging'], str(tmp_path / 'benchmark.csv'), env_builder=lambda s: EBusDroneEnv(smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}}, smoke_test=True)
    assert rows and (tmp_path / 'benchmark.csv').exists() and (tmp_path / 'benchmark.json').exists()


def test_method_name_normalization():
    assert normalize_method_name('dwell_based_greedy') == 'dwell_greedy'
    assert normalize_method_name('proposed') == 'am_dueling_ddqn_dr'


def test_uniform_method_parsing_supports_paper_durations():
    assert uniform_seconds_from_method('uniform_15') == 15
    assert uniform_seconds_from_method('uniform_30') == 30
    assert uniform_seconds_from_method('uniform_45') == 45
    assert uniform_seconds_from_method('uniform_60') == 60
    assert uniform_seconds_from_method('uniform_120') == 120


def test_build_policy_supports_all_learning_ablation_methods(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    for method in ["dqn_dr", "ddqn_dr", "am_ddqn_dr", "am_dueling_ddqn_dr"]:
        policy = build_policy(method, env, out_root=str(tmp_path), train_if_missing=True, smoke_test=True, seed=7, instance_name="small")
        assert policy is not None


def test_evaluation_full_horizon_default_not_50():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics['steps'] != 50


def test_max_steps_explicit_truncates():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=3)
    assert metrics['steps'] <= 3
    assert metrics['max_steps'] == 3
    assert metrics['full_horizon_completed'] is False


def test_full_horizon_only_true_when_operating_horizon_reached():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics['full_horizon_completed'] == (metrics['termination_reason'] == 'horizon_reached')


def test_sensitivity_changes_factor(tmp_path):
    out = tmp_path / 'sens.csv'
    rows = run_sensitivity(['no_charging'], str(out), env_builder=lambda s, c: EBusDroneEnv(config=c, smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}}, factor='passenger_intensity', values=[0.75, 1.25], smoke_test=True)
    assert {r['sensitivity_value'] for r in rows} == {0.75, 1.25}
    assert all('whether_instance_regenerated' in r for r in rows)


def test_sensitivity_regen_and_offline_rules_and_scenario_consistency(tmp_path):
    calls = {'regen': 0, 'offline': 0, 'env': []}

    def regen_cb(cfg_mod, *_args):
        calls['regen'] += 1

    def offline_cb(_cfg_mod, _instance, _seed, factor, value):
        calls['offline'] += 1
        if factor == 'locker_capacity' and value == 0.0:
            return 'infeasible'
        return 'resolved'

    def env_builder(seed, cfg_mod):
        calls['env'].append((seed, cfg_mod.get('power', {}).get('station_capacity_kw')))
        return EBusDroneEnv(config=cfg_mod, smoke_test=True)

    cfg = {'paths': {'outputs': str(tmp_path)}, '_sensitivity_hooks': {'regenerate_instance': regen_cb, 'resolve_offline': offline_cb}}

    rows_p = run_sensitivity(['no_charging', 'max_feasible'], str(tmp_path / 'p.csv'), env_builder, 'small', [3], cfg, 'passenger_intensity', [0.9], smoke_test=True)
    assert calls['regen'] == 0 and calls['offline'] == 0
    assert all(not r['whether_instance_regenerated'] and not r['whether_offline_resolved'] for r in rows_p)
    assert len({r['scenario_token'] for r in rows_p}) == 1

    rows_n = run_sensitivity(['no_charging'], str(tmp_path / 'n.csv'), env_builder, 'small', [3], cfg, 'num_customers', [1.2], smoke_test=True)
    assert calls['regen'] == 1 and calls['offline'] == 1
    assert rows_n[0]['whether_instance_regenerated'] and rows_n[0]['whether_offline_resolved']

    rows_d = run_sensitivity(['no_charging'], str(tmp_path / 'd.csv'), env_builder, 'small', [3], cfg, 'drones_per_station', [1.1], smoke_test=True)
    assert calls['offline'] == 2 and rows_d[0]['whether_offline_resolved']

    rows_pow = run_sensitivity(['no_charging'], str(tmp_path / 'pow.csv'), env_builder, 'small', [4], cfg, 'station_power_capacity', [1600.0], smoke_test=True)
    assert calls['offline'] == 2
    assert not rows_pow[0]['whether_offline_resolved'] and not rows_pow[0]['whether_instance_regenerated']
    assert any(v == 1600.0 for _s, v in calls['env'])

    rows_inf = run_sensitivity(['no_charging'], str(tmp_path / 'inf.csv'), env_builder, 'small', [5], cfg, 'locker_capacity', [0.0], smoke_test=True)
    assert rows_inf[0]['offline_status'] == 'infeasible'
    assert rows_inf[0]['full_horizon_completed'] is False
    assert rows_inf[0]['termination_reason'] == 'offline_infeasible'


def test_eval_metrics_saved_per_seed(tmp_path):
    rows=[{'method':'no_charging','instance':'small','seed':1,'total_reward':0.0},{'method':'no_charging','instance':'small','seed':2,'total_reward':0.0}]
    save_eval_metrics(rows, str(tmp_path/'eval.csv'))
    txt=(tmp_path/'eval.csv').read_text()
    assert 'seed' in txt and '2' in txt


def test_empty_rows_raise_error(tmp_path):
    with pytest.raises(ValueError):
        save_eval_metrics([], str(tmp_path/'empty.csv'))
