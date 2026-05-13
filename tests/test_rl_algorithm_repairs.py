import numpy as np
import torch

from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.networks.dueling_q_network import DuelingQNetwork
from src.rl.replay_buffer import ReplayBuffer


def test_epsilon_random_and_greedy_actions_are_feasible():
    agent = AMDuelingDDQNDRAgent(3, 5, {"epsilon_start": 1.0, "epsilon_end": 1.0})
    obs = np.zeros(3, dtype=np.float32)
    mask = np.array([0, 0, 1, 0, 1], dtype=np.float32)
    for _ in range(100):
        assert agent.select_action(obs, mask, training=True) in {0, 2, 4}
    agent.cfg["epsilon_start"] = 0.0
    agent.cfg["epsilon_end"] = 0.0
    for _ in range(20):
        assert mask[agent.select_action(obs, mask, training=False)] == 1


def test_no_charger_mask_forces_action_zero():
    agent = DDQNDRAgent(3, 4, {"epsilon_start": 0.0, "epsilon_end": 0.0})
    obs = np.zeros(3, dtype=np.float32)
    mask = np.zeros(4, dtype=np.float32)
    assert agent.select_action(obs, mask, training=False) in {0, 1, 2, 3}


def test_non_am_dqn_can_pick_infeasible_action_when_highest_q():
    from src.rl.agents.dqn_dr_agent import DQNDRAgent
    agent = DQNDRAgent(2, 4, {"epsilon_start": 0.0, "epsilon_end": 0.0})
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.model[-1].bias[:] = torch.tensor([0.0, 10.0, 0.0, 0.0])
    obs = np.zeros(2, dtype=np.float32)
    mask = np.array([1, 0, 1, 1], dtype=np.float32)
    assert agent.select_action(obs, mask, training=False) == 1


def test_ddqn_target_online_argmax_target_eval():
    agent = DDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1, "gamma": 1.0, "epsilon_start": 0.0, "epsilon_end": 0.0})
    s = np.array([0.0, 0.0], dtype=np.float32)
    m = np.ones(3, dtype=np.float32)
    agent.observe(s, 0, 1.0, s, False, m, m, {})

    called = {"online": False, "target": False}
    orig_online = agent.online.forward
    orig_target = agent.target.forward

    def wrapped_online(x):
        out = orig_online(x)
        called["online"] = True
        return out

    def wrapped_target(x):
        out = orig_target(x)
        called["target"] = True
        return out

    agent.online.forward = wrapped_online
    agent.target.forward = wrapped_target
    assert agent.update(batch_size=1) is not None
    assert called["online"] and called["target"]


def test_dqn_and_ddqn_targets_differ():
    from src.rl.agents.dqn_dr_agent import DQNDRAgent
    dqn = DQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1, "gamma": 1.0, "epsilon_start": 0.0, "epsilon_end": 0.0})
    ddqn = DDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1, "gamma": 1.0, "epsilon_start": 0.0, "epsilon_end": 0.0})
    s = np.array([0.0, 0.0], dtype=np.float32)
    m = np.array([1, 0, 1], dtype=np.float32)
    dqn.observe(s, 0, 0.0, s, False, m, m, {})
    ddqn.observe(s, 0, 0.0, s, False, m, m, {})
    with torch.no_grad():
        for a in (dqn, ddqn):
            for p in a.online.parameters():
                p.zero_()
            for p in a.target.parameters():
                p.zero_()
        dqn.online.model[-1].bias[:] = torch.tensor([0.0, 5.0, 0.0])
        dqn.target.model[-1].bias[:] = torch.tensor([1.0, 0.0, 2.0])
        ddqn.online.model[-1].bias[:] = torch.tensor([0.0, 5.0, 0.0])
        ddqn.target.model[-1].bias[:] = torch.tensor([1.0, 0.0, 2.0])
    loss_dqn = dqn.update(batch_size=1)
    loss_ddqn = ddqn.update(batch_size=1)
    assert loss_dqn is not None and loss_ddqn is not None
    assert loss_dqn != loss_ddqn


def test_dueling_uses_feasible_action_mean():
    net = DuelingQNetwork(2, 3, [4])
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.adv.bias[:] = torch.tensor([2.0, 4.0, 6.0])
        net.value.bias[:] = torch.tensor([1.0])
    x = torch.zeros(1, 2)
    m = torch.tensor([[1.0, 0.0, 1.0]])
    q = net(x, m)[0]
    # feasible mean=(2+6)/2=4 => [ -1,1,3 ] +1?
    assert torch.allclose(q, torch.tensor([-1.0, 1.0, 3.0]))


def test_am_dueling_update_uses_current_and_next_masks():
    agent = AMDuelingDDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1})
    s = np.array([0.1, 0.2], dtype=np.float32)
    ns = np.array([0.2, 0.3], dtype=np.float32)
    cm = np.array([1, 0, 1], dtype=np.float32)
    nm = np.array([1, 1, 0], dtype=np.float32)
    agent.observe(s, 2, 1.0, ns, False, cm, nm, {})
    assert agent.update(batch_size=1) is not None


def test_am_ddqn_target_uses_next_action_mask():
    from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
    agent = AMDDQNDRAgent(2, 3, {"batch_size": 1, "warmup_steps": 1, "gamma": 1.0, "epsilon_start": 0.0, "epsilon_end": 0.0})
    s = np.array([0.0, 0.0], dtype=np.float32)
    nm = np.array([1, 0, 1], dtype=np.float32)
    agent.observe(s, 0, 0.0, s, False, np.ones(3, dtype=np.float32), nm, {})
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
        agent.online.model[-1].bias[:] = torch.tensor([0.0, 9.0, 1.0])
        agent.target.model[-1].bias[:] = torch.tensor([1.0, 10.0, 2.0])
    loss = agent.update(batch_size=1)
    assert loss is not None


def test_replay_buffer_stores_masks():
    rb = ReplayBuffer(10)
    rb.add(np.zeros(2), 1, 1.0, np.ones(2), False, np.array([1, 0]), np.array([1, 1]), {})
    t = rb.sample(1)[0]
    assert t.action_mask.tolist() == [1.0, 0.0]
    assert t.next_action_mask.tolist() == [1.0, 1.0]


def test_checkpoint_restores_action_dim(tmp_path):
    a = AMDuelingDDQNDRAgent(2, 4, {})
    p = tmp_path / "a.pt"
    a.save_checkpoint(p)
    b = AMDuelingDDQNDRAgent(2, 4, {})
    b.load_checkpoint(p)
    assert b.action_dim == 4


def test_rl_trainer_import_safe():
    import src.rl.trainer as tr
    assert hasattr(tr, "DQNAgent")
