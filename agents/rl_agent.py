import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=0.001):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class RLTrainer:
    def __init__(self, env, config):
        self.env = env
        self.agent = DQNAgent(env.size*2, 4, config['agent']['learning_rate'])
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config['agent']['learning_rate'])
        self.memory = deque(maxlen=10000)
        self.epsilon = config['agent']['epsilon_start']
    def train_episode(self):
        state = self.env.reset()
        for _ in range(config['training']['max_steps_per_episode']):
            if random.random() < self.epsilon:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    qvals = self.agent(torch.FloatTensor(state))
                    action = qvals.argmax().item()
            next_state, reward, done, _ = self.env.step(action)