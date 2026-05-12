import numpy as np, torch
from src.rl.networks.q_network import QNetwork
from src.rl.networks.dueling_q_network import DuelingQNetwork
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent


def test_network_shapes_and_mask():
    q=QNetwork(4,9); x=torch.randn(2,4); assert q(x).shape==(2,9)
    d=DuelingQNetwork(4,9); m=torch.ones(2,9); assert d(x).shape==(2,9); assert d(x,m).shape==(2,9)


def test_agents_return_action_and_masked_feasible():
    obs=np.zeros(4,dtype=np.float32); mask=np.array([1,0,0,1,0,0,0,0,0],dtype=np.float32)
    for cls in [DQNDRAgent,DDQNDRAgent,AMDDQNDRAgent,AMDuelingDDQNDRAgent]:
        a=cls(4,9,{}).select_action(obs,mask,training=False)
        assert isinstance(a,int)
    a=AMDDQNDRAgent(4,9,{}).select_action(obs,mask,training=False)
    assert mask[a]==1
