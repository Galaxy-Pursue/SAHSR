import torch
from torch import nn
import torch.nn.functional as F
# from utils import seed_torch

class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=2) / self.temperature

        # 创建掩码，标识正样本
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # 提取正样本和负样本的相似度
        positive_samples = similarity_matrix[mask].view(n, -1)
        negative_samples = similarity_matrix[~mask].view(n, -1)

        # 正负样本相似度归一化
        positive_samples = torch.exp(positive_samples)
        negative_samples = torch.exp(negative_samples).sum(dim=1, keepdim=True)

        # InfoNCE loss
        loss = (-torch.log(positive_samples / (positive_samples + negative_samples))).mean()

        return loss

class CSLoss2(nn.Module):
    def __init__(self, k_size, margin1=0, margin2=0.7, temperature=0.5):
        super(CSLoss2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin2)
        self.info_fn = InfoNCELoss(temperature)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)

        dist_pc = (inputs - centers) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()
        dist_pc = (dist_pc - self.margin1).clamp(min=0.0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n // 2, self.k_size // 2): # total p 个
            dist_an.append((self.margin2 - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
        dist_an = torch.stack(dist_an)

        centers_uni = centers[:n//2:self.k_size//2]
        labels_uni = torch.arange(0, centers_uni.shape[0], 1, dtype=centers_uni.dtype, device=centers_uni.device)

        dist_an_mean = self.info_fn(centers_uni, labels_uni)

        # Compute ranking hinge loss
        # y = dist_an.data.new()
        # y.resize_as_(dist_an.data)
        # y.fill_(1)
        loss = dist_pc.mean() + dist_an_mean # dist_an.mean()
        return loss, dist_pc.mean(), dist_an_mean, dist_an.mean()


class HCLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID """

    def __init__(self, margin=0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
            Args:
            - inputs: feature with shape (batch_size, feat_dim)
            - labels: ground truth labels with shape (batch_size)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


if __name__ == "__main__":
    seed_torch(1234)
    imgs = torch.randn(80, 2048)
    cam_ids = torch.tensor([3] * 40 + [2] * 40)
    labels = torch.tensor([338, 338, 338, 338, 115, 115, 115, 115, 241, 241, 241, 241, 291, 291,
                           291, 291, 58, 58, 58, 58, 10, 10, 10, 10, 110, 110, 110, 110,
                           335, 335, 335, 335, 234, 234, 234, 234, 332, 332, 332, 332] * 2)

    loss_fn = CSLoss2(k_size=8)
    print(loss_fn(imgs, labels))