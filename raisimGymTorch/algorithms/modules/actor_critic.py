import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, architecture, distribution, device='gpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device


