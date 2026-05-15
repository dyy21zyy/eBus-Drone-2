from __future__ import annotations

import inspect

import numpy as np
import torch


class LearnedPolicy:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, observation, action_mask, info=None) -> int:
        action = self._call_eval_action(observation, action_mask, info)
        mask = np.asarray(action_mask)
        if action < 0 or action >= int(mask.size) or mask[action] <= 0:
            return self._masked_greedy_action(observation, action_mask)
        return int(action)

    def _call_eval_action(self, observation, action_mask, info=None) -> int:
        fn = self.agent.select_action
        try:
            sig = inspect.signature(fn)
            if "training" in sig.parameters:
                return int(fn(observation, action_mask, training=False))
        except (TypeError, ValueError):
            pass
        try:
            return int(fn(observation, action_mask, training=False))
        except TypeError:
            pass
        try:
            return int(fn(observation, action_mask, False))
        except TypeError:
            pass
        try:
            return int(fn(observation, action_mask))
        except TypeError:
            return int(fn(observation, action_mask, info))

    def _masked_greedy_action(self, observation, action_mask) -> int:
        if hasattr(self.agent, "online"):
            with torch.no_grad():
                device = getattr(self.agent, "device", "cpu")
                obs = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.float32, device=device)
                if mask.numel() == 0:
                    raise ValueError("Action mask is empty.")
                if torch.all(mask <= 0):
                    mask = mask.clone()
                    mask[0] = 1.0
                try:
                    q = self.agent.online(obs, mask.unsqueeze(0))[0]
                except TypeError:
                    q = self.agent.online(obs)[0]
                q = q.masked_fill(mask <= 0, -1e9)
                return int(torch.argmax(q).item())
        feasible = np.flatnonzero(np.asarray(action_mask) > 0)
        return int(feasible[0]) if feasible.size else 0

    def act(self, observation, action_mask) -> int:
        return self.select_action(observation, action_mask)
