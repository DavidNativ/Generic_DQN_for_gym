import torch
import numpy as np

class Control:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, Q, epsilon, training):
        if not training:
            return self.greedy(Q)

        #epsilon greedy
        if np.random.random() < epsilon:
            return self.random()
        else:
            return self.greedy(Q)

    def random(self):
        return np.random.choice(self.action_space)

    def greedy(self, Q):
        return torch.argmax(Q, dim=-1).item()
