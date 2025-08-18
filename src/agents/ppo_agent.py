import torch
import torch.nn as nn
import torch.distributions as D

class ActorCritic(nn.Module):
    """
    Minimal actor–critic for continuous 2D actions.
    - Input: full global observation (obs_dim)
    - Output: 2D action and a state value
    """
    def __init__(self, obs_dim: int, act_dim: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.pi_mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learned log-std
        self.v_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.pi_mean(h)
        std = self.log_std.exp().expand_as(mean)
        dist = D.Normal(mean, std)
        value = self.v_head(h).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, value = self.forward(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value
