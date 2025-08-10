import torch
import torch.nn as nn
import torch.distributions as D

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_std.exp()

    def get_action(self, x):
        mu, std = self(x)
        dist = D.Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1)