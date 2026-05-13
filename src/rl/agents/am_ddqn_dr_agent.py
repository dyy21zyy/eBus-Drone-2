from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.agents.ddqn_dr_agent import DDQNDRAgent


class AMDDQNDRAgent(DDQNDRAgent):
    def _feasible(self, mask):
        feasible = [i for i, v in enumerate(mask) if v > 0]
        if not feasible:
            feasible = [0]
        return sorted(set(feasible))

    def select_action(self, observation, action_mask, training=True) -> int:
        return super().select_action(observation, action_mask, training=training)

    def update(self, batch_size=None):
        bs = batch_size or int(self.cfg.get("batch_size", 32))
        if len(self.buffer) < max(bs, int(self.cfg.get("warmup_steps", 20))):
            return None
        b = self.buffer.sample(bs)
        obs = torch.tensor(np.stack([t.observation for t in b]), dtype=torch.float32, device=self.device)
        act = torch.tensor([t.action_index for t in b], dtype=torch.long, device=self.device)
        rew = torch.tensor([t.reward for t in b], dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.stack([t.next_observation for t in b]), dtype=torch.float32, device=self.device)
        nmask = torch.tensor(np.stack([t.next_action_mask for t in b]), dtype=torch.float32, device=self.device)
        nmask = nmask.clone()
        all_zero = torch.all(nmask <= 0, dim=1)
        nmask[all_zero, 0] = 1.0
        done = torch.tensor([t.done for t in b], dtype=torch.float32, device=self.device)
        q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_on = self.online(nxt).masked_fill(nmask <= 0, -1e9)
            a_next = q_on.argmax(dim=1)
            q_next = self.target(nxt).gather(1, a_next.unsqueeze(1)).squeeze(1)
            y = rew + (1 - done) * float(self.cfg.get("gamma", 0.99)) * q_next
        loss = F.mse_loss(q, y)
        self.optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.online.parameters(), float(self.cfg.get("gradient_clip_norm", 10.0))); self.optim.step(); self.steps += 1
        self._target_update()
        return float(loss.item())
