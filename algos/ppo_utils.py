import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, MultivariateNormal
import numpy as np

class Policy(torch.nn.Module):
        
    def __init__(self, state_space, action_space, hidden_size=32, base_logstd=1):
            super().__init__()
            self.state_space = state_space
            self.action_space = action_space

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
            self.actor_logstd = base_logstd * torch.ones(action_space)
            
            self.init_weights()

            self.critic.apply(self.init_weights)
            self.actor_mean.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) is torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 1e-1)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        mean = self.fc3_a(x_a)
        logstd = self.actor_logstd.expand_as(mean)
        
        action_dist = MultivariateNormal(mean, torch.diag_embed(torch.exp(logstd)))


        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)

        #x_a = torch.mean(x_a, dim=0, keepdim=True)
        # if len(x_a.shape) > 1:
        #     x_a = torch.mean(x_a, dim=0)
        # action_logstd = self.actor_logstd.expand_as(x_a)
        # #action_probs = F.softmax(x_a, dim=-1)
        # actor_std = torch.exp(action_logstd)
        # #print(x_a.shape)
        # try:
        #     cov_matrix = torch.mul(actor_std, torch.eye(self.action_space))
        # except Exception as e:
        #     print(e, x_a.shape, actor_std.shape, self.action_space), 
        # action_dist = MultivariateNormal(x_a, cov_matrix)

        
        return action_dist, x_c
    
    def set_logstd_ratio(self, ratio_of_episodes):
        self.actor_logstd = torch.ones(self.action_space) * self.base_logstd * ratio_of_episodes
