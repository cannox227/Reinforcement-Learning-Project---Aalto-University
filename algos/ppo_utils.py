import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, MultivariateNormal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32, base_logstd=1):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_space, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(state_space, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space)
        )

        self.base_logstd = base_logstd
        self.actor_logstd = torch.ones(action_space) * base_logstd

        self.critic.apply(self.init_weights)
        self.actor_mean.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) is torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 1e-1)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_matrix = torch.mul(torch.eye(self.action_space), action_std)
        action_dist = MultivariateNormal(action_mean, cov_matrix)
        return action_dist, self.critic(x)
    
    def set_logstd_ratio(self, ratio_of_episodes):
        self.actor_logstd = torch.ones(1, self.action_space) * self.base_logstd * ratio_of_episodes