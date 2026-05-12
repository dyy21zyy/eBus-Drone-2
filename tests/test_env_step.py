from src.data_generation.scenario_generator import generate_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.env.ebus_drone_env import EBusDroneEnv
from src.utils.config import load_yaml


def test_env_uses_loaded_generated_data():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {"passenger": {"passenger_arrivals": {s['stop_id']: [0] for s in instance['network']['stops']}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    assert env.instance is not None
    assert env.scenario is not None
    assert env.assignment is not None
    assert env.state['horizon'] == instance['horizon_minutes']
    assert env.state['battery_max'] == instance['bus']['battery_capacity_kwh']


def test_cli_eval_loads_real_data_path(monkeypatch):
    from src import main as main_mod
    called = {"value": False}

    def _fake_env(**kwargs):
        assert kwargs.get("instance") is not None
        assert kwargs.get("scenario") is not None
        assert kwargs.get("assignment") is not None
        called["value"] = True
        class Dummy:
            pass
        return Dummy()

    monkeypatch.setattr(main_mod, "EBusDroneEnv", _fake_env)
    monkeypatch.setattr(main_mod, "evaluate_policy", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(main_mod, "build_policy", lambda method: object())
    import sys
    sys.argv = ["prog", "--mode", "eval", "--config", "configs/default.yaml", "--instance", "small", "--method", "no_charging", "--seed", "1", "--smoke-test"]
    main_mod.main()
    assert called["value"] is True
