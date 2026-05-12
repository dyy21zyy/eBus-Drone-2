from __future__ import annotations


class LearnedPolicy:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, observation, action_mask, info=None) -> int:
        return int(self.agent.select_action(observation, action_mask, info))

    def act(self, observation, action_mask) -> int:
        return self.select_action(observation, action_mask)
