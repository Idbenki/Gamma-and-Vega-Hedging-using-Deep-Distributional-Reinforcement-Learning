import torch
import torch.nn as nn
import torch.nn.functional as F


def build_actor(state_dim=5, hidden=256, n_layers=3):
    layers = []
    in_dim = state_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())  # action in [0,1]
    return nn.Sequential(*layers)


class CriticQuantile(nn.Module):
    def __init__(self, state_dim=5, action_dim=1, M=100, h1=512, h2=512, h3=256):
        super().__init__()
        self.M = int(M)
        in_dim = int(state_dim) + int(action_dim)
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, self.M)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def build_critic_quantile(state_dim=5, action_dim=1, M=100, h1=512, h2=512, h3=256):
    return CriticQuantile(state_dim=state_dim, action_dim=action_dim, M=M, h1=h1, h2=h2, h3=h3)


def actor_forward(actor, state):
    return actor(state)


def critic_forward(critic, state, action):
    return critic(state, action)
