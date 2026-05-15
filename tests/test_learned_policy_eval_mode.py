import numpy as np
import torch

from src.policies.learned_policy import LearnedPolicy


class DummyAgentTrainingFlag:
    def __init__(self):
        self.training_args = []

    def select_action(self, observation, action_mask, training=True):
        self.training_args.append(training)
        return 0


class DummyAgentIgnoresMask:
    def __init__(self):
        self.device = torch.device("cpu")

        class Net(torch.nn.Module):
            def forward(self, obs):
                del obs
                return torch.tensor([[0.1, 5.0, 1.0]], dtype=torch.float32)

        self.online = Net()

    def select_action(self, observation, action_mask, training=True):
        del observation, action_mask, training
        return 1


def test_learned_policy_calls_training_false_during_eval():
    agent = DummyAgentTrainingFlag()
    policy = LearnedPolicy(agent)

    action = policy.select_action(np.array([0.0]), np.array([1, 1, 1]), info={"foo": "bar"})

    assert action == 0
    assert agent.training_args == [False]


def test_learned_policy_enforces_masked_greedy_when_agent_returns_infeasible():
    agent = DummyAgentIgnoresMask()
    policy = LearnedPolicy(agent)
    obs = np.array([0.0], dtype=np.float32)
    mask = np.array([1, 0, 1], dtype=np.float32)

    a1 = policy.select_action(obs, mask)
    a2 = policy.select_action(obs, mask)

    assert a1 == 2
    assert a2 == 2
