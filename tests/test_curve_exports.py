from pathlib import Path
import subprocess
import pandas as pd


def run(cmd: str):
    subprocess.run(cmd, shell=True, check=True)


def test_curve_exports_smoke(tmp_path):
    out = tmp_path / 'out'
    log_path = tmp_path / "run.log"
    run(f"PYTHONUNBUFFERED=1 python -u -m src.main --mode pipeline --config configs/debug.yaml --instance small --seeds 1 --methods uniform battery_threshold proposed --smoke --output-dir {out} --overwrite --train-if-missing --log-interval 1 > {log_path} 2>&1")
    # training curve for rl
    train_csv = out / 'smoke' / 'curves' / 'train' / 'small' / 'am_dueling_ddqn_dr_seed_1_train_curve.csv'
    eval_csv = out / 'smoke' / 'curves' / 'eval' / 'small' / 'am_dueling_ddqn_dr_seed_1_eval_curve.csv'
    eval_cum_csv = out / 'smoke' / 'curves' / 'eval' / 'small' / 'am_dueling_ddqn_dr_seed_1_eval_cumulative_curve.csv'
    assert train_csv.exists() and train_csv.stat().st_size > 0
    assert eval_csv.exists() and eval_csv.stat().st_size > 0
    assert eval_cum_csv.exists() and eval_cum_csv.stat().st_size > 0
    tdf = pd.read_csv(train_csv)
    edf = pd.read_csv(eval_csv)
    assert {'episode','method','instance','seed','steps'}.issubset(tdf.columns)
    assert {'eval_episode','method','instance','seed','total_cost','total_reward'}.issubset(edf.columns)
    assert tdf['episode'].is_monotonic_increasing
    assert edf['eval_episode'].is_monotonic_increasing
    assert '[train][episode]' in log_path.read_text(encoding='utf-8')
    assert '[pipeline]' in log_path.read_text(encoding='utf-8')
    fig = out / 'smoke' / 'figures' / 'eval' / 'small' / 'am_dueling_ddqn_dr_seed_1_eval_total_cost_curve.png'
    assert fig.exists() and fig.stat().st_size > 0
    train_fig = out / 'smoke' / 'figures' / 'train' / 'small' / 'am_dueling_ddqn_dr_seed_1_training_reward_curve.png'
    assert train_fig.exists() and train_fig.stat().st_size > 0


def test_log_interval_zero_disables_episode_print(tmp_path):
    out = tmp_path / "out2"
    log_path = tmp_path / "run2.log"
    run(f"PYTHONUNBUFFERED=1 python -u -m src.main --mode pipeline --config configs/debug.yaml --instance small --seeds 1 --methods am_dueling_ddqn_dr --smoke --train-if-missing --output-dir {out} --overwrite --log-interval 0 > {log_path} 2>&1")
    txt = log_path.read_text(encoding="utf-8")
    assert "[train][episode]" not in txt


def test_curve_aggregation_and_alias(tmp_path):
    from src.harness.curve_export import aggregate_curves
    base = tmp_path / 'o' / 'curves' / 'eval' / 'small'
    base.mkdir(parents=True)
    for sd in [1,2]:
        pd.DataFrame([
            {'eval_episode':1,'total_cost':10+sd,'total_reward':-1*sd,'minimum_bus_battery':50,'battery_safety_violation_count':0},
            {'eval_episode':2,'total_cost':12+sd,'total_reward':-2*sd,'minimum_bus_battery':49,'battery_safety_violation_count':1},
        ]).to_csv(base / f'am_dueling_ddqn_dr_seed_{sd}_eval_curve.csv', index=False)
    p = aggregate_curves(str(tmp_path / 'o'), 'small', 'am_dueling_ddqn_dr', 'eval')
    assert p and Path(p).exists()
    adf = pd.read_csv(p)
    assert {'eval_episode','mean_total_cost','std_total_cost'}.issubset(adf.columns)
