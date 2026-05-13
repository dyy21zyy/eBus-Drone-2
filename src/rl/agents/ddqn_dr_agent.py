from src.rl.agents.dqn_dr_agent import DQNDRAgent
import torch


class DDQNDRAgent(DQNDRAgent):
    def update(self, batch_size=None):
        bs = batch_size or int(self.cfg.get('batch_size', 32))
        if len(self.buffer) < max(bs, int(self.cfg.get('warmup_steps', 20))):
            return None
        b = self.buffer.sample(bs)
        import numpy as np, torch.nn.functional as F
        obs = torch.tensor(np.stack([t.observation for t in b]), dtype=torch.float32, device=self.device)
        act = torch.tensor([t.action_index for t in b], dtype=torch.long, device=self.device)
        rew = torch.tensor([t.reward for t in b], dtype=torch.float32, device=self.device)
        nxt = torch.tensor(np.stack([t.next_observation for t in b]), dtype=torch.float32, device=self.device)
        done = torch.tensor([t.done for t in b], dtype=torch.float32, device=self.device)
        q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = self.online(nxt).argmax(dim=1)
            q_next = self.target(nxt).gather(1, a_next.unsqueeze(1)).squeeze(1)
            y = rew + (1 - done) * float(self.cfg.get('gamma', 0.99)) * q_next
        loss = F.mse_loss(q, y)
        self.optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.online.parameters(), float(self.cfg.get('gradient_clip_norm', 10.0))); self.optim.step(); self.steps += 1
        self._target_update()
        return float(loss.item())
