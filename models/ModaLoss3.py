import torch
from torch import nn
from torch.nn import functional as F


class ModaReduceLoss(nn.Module):
    def __init__(self):
        super(ModaReduceLoss, self).__init__()
        # self.args = args
        # self.loss = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')

    def distance_matrix(self, x, y):
        # 计算两组特征之间的距离矩阵
        dist_m = torch.cdist(x, y, p=2)
        return dist_m

    def forward(self, features, ids, sub):
        # 计算两个模态特征之间的距离矩阵
        features_mod1 = features[sub]
        features_mod2 = features[~sub]
        dist_m1 = self.distance_matrix(features_mod1, features_mod1)  # 计算第一个模态的距离矩阵
        dist_m2 = self.distance_matrix(features_mod2, features_mod2)  # 计算第二个模态的距离矩阵
        dist_cross = self.distance_matrix(features_mod1, features_mod2)

        # 将距离矩阵转化为概率分布
        P = F.softmax(-dist_m1, dim=1)
        Q = F.softmax(-dist_m2, dim=1)

        # 计算KL散度

        kl_loss_12 = self.KL(P.log(), Q)
        kl_loss_21 = self.KL(Q.log(), P)

        # 计算模态间的KL散度
        cross_P = F.softmax(-dist_cross, dim=1)
        cross_Q = F.softmax(-dist_cross.t(), dim=1)
        cross_kl_loss_12 = self.KL(cross_P.log(), cross_Q)
        cross_kl_loss_21 = self.KL(cross_Q.log(), cross_P)

        bg_loss_kl = kl_loss_12 + kl_loss_21
        return kl_loss_12, bg_loss_kl, cross_kl_loss_12, cross_kl_loss_21

        # loss = self.loss(output, target)