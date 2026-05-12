from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.network_factory import QNetwork
from src.rl.replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    method: str = "proposed"
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    target_update: int = 25
    warmup_steps: int = 20
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    capacity: int = 5000


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AgentConfig):
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = cfg.method.lower()
        self.use_double = m in {"ddqn_dr", "am_ddqn_dr", "proposed", "am_dueling_ddqn_dr"}
        self.use_mask = m in {"am_ddqn_dr", "proposed", "am_dueling_ddqn_dr"}
        dueling = m in {"proposed", "am_dueling_ddqn_dr"}
        self.online = QNetwork(obs_dim, action_dim, dueling=dueling).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.capacity)
        self.step_count = 0

    def epsilon(self) -> float:
        p = min(1.0, self.step_count / max(1, self.cfg.epsilon_decay))
        return self.cfg.epsilon_start + p * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, observation, action_mask, info=None) -> int:
        if random.random() < self.epsilon():
            feasible = [i for i, v in enumerate(action_mask) if v > 0.5]
            return random.choice(feasible) if (feasible and self.use_mask) else random.randrange(self.action_dim)
        with torch.no_grad():
            obs_t = torch.tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(np.asarray(action_mask), dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(obs_t, mask_t if self.use_mask else None)[0]
            if self.use_mask:
                q = q.masked_fill(mask_t[0] <= 0, -1e9)
            return int(torch.argmax(q).item())

    def update(self):
        if len(self.buffer) < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        obs = torch.tensor(np.stack([t.observation for t in batch]), dtype=torch.float32, device=self.device)
        act = torch.tensor([t.action_index for t in batch], dtype=torch.long, device=self.device)
        rew = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.stack([t.next_observation for t in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        mask = torch.tensor(np.stack([t.action_mask for t in batch]), dtype=torch.float32, device=self.device)
        nxt_mask = torch.tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.float32, device=self.device)
        q = self.online(obs, mask if self.use_mask else None).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_online = self.online(nxt, nxt_mask if self.use_mask else None)
            if self.use_mask:
                q_next_online = q_next_online.masked_fill(nxt_mask <= 0, -1e9)
            next_a = q_next_online.argmax(dim=1)
            if self.use_double:
                q_next_target_all = self.target(nxt, nxt_mask if self.use_mask else None)
                q_next = q_next_target_all.gather(1, next_a.unsqueeze(1)).squeeze(1)
            else:
                q_next_target_all = self.target(nxt, nxt_mask if self.use_mask else None)
                if self.use_mask:
                    q_next_target_all = q_next_target_all.masked_fill(nxt_mask <= 0, -1e9)
                q_next = q_next_target_all.max(dim=1).values
            y = rew + (1.0 - done) * self.cfg.gamma * q_next
        loss = F.mse_loss(q, y)
        self.optim.zero_grad(); loss.backward(); self.optim.step()
        self.step_count += 1
        if self.step_count % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
