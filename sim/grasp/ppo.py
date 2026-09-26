"""A compact PPO for the grasp environment: MLP actor-critic, GAE, clipped loss.

Kept dependency-free beyond torch and numpy so it runs in the lerobot venv
on the Mac (CPU or MPS) and on a rented box alike. `Policy` is what
`train.py` saves and `eval.py` loads: weights plus the observation
normaliser, in one file.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningNorm:
    """Welford mean/var of the observations, frozen at evaluation time."""

    def __init__(self, dim, eps=1e-4):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.size)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((np.asarray(x, np.float64) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load(self, s):
        self.mean, self.var, self.count = np.asarray(s["mean"]), np.asarray(s["var"]), float(s["count"])


def mlp(inp, out, hidden=(256, 256), gain=1.0):
    layers, last = [], inp
    for h in hidden:
        lin = nn.Linear(last, h); nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]; last = h
    lin = nn.Linear(last, out); nn.init.orthogonal_(lin.weight, gain); nn.init.zeros_(lin.bias)
    return nn.Sequential(*layers, lin)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden, gain=0.01)
        self.v = mlp(obs_dim, 1, hidden, gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        mu = self.pi(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.v(obs).squeeze(-1)


class Policy:
    """Actor-critic plus normaliser; `act` is what the environments call."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), device="cpu"):
        self.net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.norm = RunningNorm(obs_dim)
        self.device = device
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, tuple(hidden)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = torch.as_tensor(self.norm(obs), device=self.device)
        d = self.net.dist(x)
        a = d.mean if deterministic else d.sample()
        return a.cpu().numpy(), d.log_prob(a).sum(-1).cpu().numpy(), self.net.value(x).cpu().numpy()

    def save(self, path):
        torch.save(dict(net=self.net.state_dict(), norm=self.norm.state(), obs_dim=self.obs_dim,
                        act_dim=self.act_dim, hidden=self.hidden), path)

    @classmethod
    def load(cls, path, device="cpu"):
        s = torch.load(path, map_location=device, weights_only=False)
        p = cls(s["obs_dim"], s["act_dim"], s["hidden"], device)
        p.net.load_state_dict(s["net"]); p.norm.load(s["norm"])
        return p


def gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards/values/dones: (T, N). Returns advantages and returns (T, N)."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def ppo_update(policy, obs, act, logp_old, adv, ret, *, epochs=10, minibatch=2048, lr=3e-4,
               clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad=0.5, opt=None):
    """One PPO update on flattened rollout arrays. Returns stats."""
    dev = policy.device
    obs_t = torch.as_tensor(policy.norm(obs), device=dev)
    act_t = torch.as_tensor(act, device=dev)
    logp_t = torch.as_tensor(logp_old, device=dev)
    adv_t = torch.as_tensor(adv, device=dev)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.as_tensor(ret, device=dev)
    if opt is None:
        opt = torch.optim.Adam(policy.net.parameters(), lr=lr, eps=1e-5)
    n = obs_t.shape[0]
    stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, kl=0.0, clipfrac=0.0)
    count = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            d = policy.net.dist(obs_t[idx])
            logp = d.log_prob(act_t[idx]).sum(-1)
            ratio = (logp - logp_t[idx]).exp()
            a = adv_t[idx]
            pi_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = 0.5 * (policy.net.value(obs_t[idx]) - ret_t[idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.net.parameters(), max_grad)
            opt.step()
            with torch.no_grad():
                stats["pi_loss"] += float(pi_loss); stats["v_loss"] += float(v_loss)
                stats["entropy"] += float(ent); stats["kl"] += float((logp_t[idx] - logp).mean())
                stats["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
            count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}, opt
