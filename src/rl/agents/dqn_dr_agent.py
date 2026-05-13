from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.network_factory import build_network
from src.rl.replay_buffer import ReplayBuffer


class DQNDRAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        self.cfg = dict(cfg)
        self.action_dim = int(action_dim)
        device_cfg = self.cfg.get("device", "auto")
        if device_cfg == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_cfg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        else:
            resolved = device_cfg
        self.device = torch.device(resolved)

        hidden = self.cfg.get("hidden_layers", [128, 128])
        self.online = build_network(obs_dim, self.action_dim, dueling=False, hidden_dims=hidden).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=float(self.cfg.get("learning_rate", self.cfg.get("lr", 1e-3))))
        self.buffer = ReplayBuffer(int(self.cfg.get("replay_buffer_size", self.cfg.get("capacity", 5000))))
        self.steps = 0
        # DQN-DR baseline intentionally does not use action masking inside the agent.
        # Environment-side repair/handling is responsible for infeasible actions.
        self.use_action_mask = False

    def _eps(self):
        s = float(self.cfg.get("epsilon_start", 1.0))
        e = float(self.cfg.get("epsilon_end", 0.05))
        d = max(1, int(self.cfg.get("epsilon_decay_steps", self.cfg.get("epsilon_decay", 500))))
        p = min(1.0, self.steps / d)
        return float(s + p * (e - s))

    def _mask_tensor(self, action_mask):
        if action_mask is None:
            m = torch.ones(self.action_dim, dtype=torch.float32, device=self.device)
        else:
            m = torch.as_tensor(np.asarray(action_mask), dtype=torch.float32, device=self.device)
        if m.numel() != self.action_dim:
            raise ValueError(f"Action mask dim {m.numel()} does not match action_dim={self.action_dim}")
        if torch.all(m <= 0):
            m = m.clone(); m[0] = 1.0
        return m

    def _feasible_indices(self, action_mask):
        m = self._mask_tensor(action_mask)
        return torch.where(m > 0)[0].detach().cpu().tolist()

    def select_action(self, observation, action_mask, training=True) -> int:
        if bool(training) and random.random() < self._eps():
            if self.use_action_mask:
                feasible = self._feasible_indices(action_mask)
                return int(random.choice(feasible))
            return int(random.randrange(self.action_dim))
        with torch.no_grad():
            q = self.online(torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            if self.use_action_mask:
                m = self._mask_tensor(action_mask)
                q = q.masked_fill(m <= 0, -1e9)
        return int(torch.argmax(q).item())

    act = select_action

    def observe(self, *args, **kwargs):
        self.buffer.add(*args, **kwargs)

    def _target_update(self):
        t = self.cfg.get("target_update_type", "hard")
        if t == "polyak":
            tau = float(self.cfg.get("polyak_tau", 0.005))
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * op.data)
        elif self.steps % int(self.cfg.get("target_update_interval", self.cfg.get("target_update", 25))) == 0:
            self.target.load_state_dict(self.online.state_dict())

    def update(self, batch_size: int | None = None):
        bs = batch_size or int(self.cfg.get("batch_size", 32))
        if len(self.buffer) < max(bs, int(self.cfg.get("warmup_steps", 20))):
            return None
        b = self.buffer.sample(bs)
        obs = torch.tensor(np.stack([t.observation for t in b]), dtype=torch.float32, device=self.device)
        act = torch.tensor([t.action_index for t in b], dtype=torch.long, device=self.device)
        rew = torch.tensor([t.reward for t in b], dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.stack([t.next_observation for t in b]), dtype=torch.float32, device=self.device)
        done = torch.tensor([t.done for t in b], dtype=torch.float32, device=self.device)
        q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_all = self.target(nxt)
            if self.use_action_mask:
                nmask = torch.tensor(np.stack([t.next_action_mask for t in b]), dtype=torch.float32, device=self.device)
                nmask = nmask.clone()
                all_zero = torch.all(nmask <= 0, dim=1)
                nmask[all_zero, 0] = 1.0
                q_next_all = q_next_all.masked_fill(nmask <= 0, -1e9)
            q_next = q_next_all.max(dim=1).values
            y = rew + (1 - done) * float(self.cfg.get("gamma", 0.99)) * q_next
        loss = F.mse_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), float(self.cfg.get("gradient_clip_norm", 10.0)))
        self.optim.step()
        self.steps += 1
        self._target_update()
        return float(loss.item())

    def checkpoint_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optim.state_dict(),
            "steps": self.steps,
            "config": self.cfg,
            "action_dim": self.action_dim,
            "action_set": self.cfg.get("action_set_seconds"),
            "normalization": self.cfg.get("normalization", {}),
        }

    def save_checkpoint(self, path):
        torch.save(self.checkpoint_dict(), path)

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck.get("target", ck["online"]))
        if "optimizer" in ck:
            self.optim.load_state_dict(ck["optimizer"])
        self.steps = int(ck.get("steps", 0))
        self.action_dim = int(ck.get("action_dim", self.action_dim))
