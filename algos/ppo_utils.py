import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, MultivariateNormal
import numpy as np
import  torch

class Policy(torch.nn.Module):
        
    def __init__(self, state_space, action_space, hidden_size=32):
            super().__init__()
            self.state_space = state_space
            self.action_space = action_space

            self.fc1_a = torch.nn.Linear(state_space, hidden_size)
            self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3_a = torch.nn.Linear(hidden_size, action_space)

            self.fc1_c = torch.nn.Linear(state_space, hidden_size)
            self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3_c = torch.nn.Linear(hidden_size, 1)

            self.actor_logstd = torch.nn.Parameter(torch.Tensor([0.0]))
            
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)

        action_probs = F.softmax(x_a, dim=-1)
        # TODO: in origin it was Categorical destribution, check it out
        actor_std = torch.exp(self.actor_logstd)
        cov_matrix = torch.mul(torch.eye(self.action_space), actor_std)
        action_dist = MultivariateNormal(action_probs, cov_matrix)

        return action_dist, x_c