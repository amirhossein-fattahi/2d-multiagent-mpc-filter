# src/agents/ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)

        # Heads
        self.mu_head = nn.Linear(last, act_dim)  # mean of Gaussian
        self.v_head  = nn.Linear(last, 1)        # value

        # Log-std per action dim (start ~ exp(-0.5) ≈ 0.61)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        # ---- IMPORTANT: initialization ----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Make *initial* actor mean exactly zero so actions are centered
        nn.init.constant_(self.mu_head.weight, 0.0)
        nn.init.constant_(self.mu_head.bias, 0.0)

        # Small value head init is fine (already near 0)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, obs):
        h = self.body(obs)
        mu = self.mu_head(h)  # unconstrained mean
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        v = self.v_head(h).squeeze(-1)
        return dist, v

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        # obs: tensor shape [1, obs_dim] (batch of 1)
        dist, v = self.forward(obs)
        if deterministic:
            action = dist.mean
            logp = torch.zeros(1, device=obs.device)
        else:
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1, keepdim=False)
        # Env expects actions in [-1, 1]; keep tanh or simple clamp.
        action = torch.tanh(action)  # nicely keeps it in [-1,1]
        return action, logp, v
