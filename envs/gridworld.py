import numpy as np

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.reset()
    def reset(self):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos
    def step(self, action):
        moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        self.agent_pos += moves[action]
        self.agent_pos = np.clip(self.agent_pos, 0, self.size-1)
        reward = -1
        done = False
        return self.agent_pos, reward, done, {}