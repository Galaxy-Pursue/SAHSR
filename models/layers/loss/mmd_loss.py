import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class ModaAlignLoss(nn.Module):
    
    def __init__(self, proto_num):
        super().__init__()
        self.proto_num = proto_num
        self.mmdloss = MMDLoss()
    
    def forward(self,moda_vis,moda_inf,moda_aux):
        assert moda_vis.shape[0] == self.proto_num
        assert moda_inf.shape[0] == self.proto_num
        assert moda_aux.shape[0] == self.proto_num
        for i in range(self.proto_num):
            if i == 0:
                loss = self.mmdloss(moda_vis[i],moda_aux[i]) + self.mmdloss(moda_inf[i],moda_aux[i])
            else:
                loss += (self.mmdloss(moda_vis[i],moda_aux[i]) + self.mmdloss(moda_inf[i],moda_aux[i]))
        return loss
        