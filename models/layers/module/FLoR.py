from torch import nn 
import torch 

class FLoR2d(nn.Module):
    def __init__(self, dim, delta = 0.5):
        super(FLoR2d, self).__init__()
        self.delta = nn.Parameter(torch.tensor(delta, dtype=torch.float))
        self.BN = nn.BatchNorm2d(dim)
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)
    def forward(self, x):
        return self.BN(x) * self.delta + self.IN(x) * (1 - self.delta)
    
    
class FLoR1d(nn.Module):
    def __init__(self, dim, delta = 0.5):
        super(FLoR1d, self).__init__()
        self.delta = nn.Parameter(torch.tensor(delta, dtype=torch.float))
        self.BN = nn.BatchNorm1d(dim)
        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)
    def forward(self, x):
        return self.BN(x) * self.delta + self.IN(x) * (1 - self.delta)