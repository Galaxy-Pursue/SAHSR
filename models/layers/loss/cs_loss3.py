import torch
from torch import nn

class ImprovedCSLoss(nn.Module):
    def __init__(self, k_size, margin1=0, margin2=0.7, margin3=0.5):
        super(ImprovedCSLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin2)
        self.contrastive_loss = nn.MarginRankingLoss(margin=margin3)

    def forward(self, inputs, targets, cam_ids):
        sub = (cam_ids == 3) + (cam_ids == 6)
        inputs1 = inputs[sub]
        inputs2 = inputs[~sub]
        assert torch.equal(targets[sub], targets[~sub])
        targets = targets[sub]
        n = inputs1.size(0)

        # Compute centers for inputs1 and inputs2
        label_uni = targets.unique()
        label_map = {label.item(): idx for idx, label in enumerate(label_uni)}

        centers1, centers2 = [], []
        for label in label_uni:
            centers1.append(inputs1[targets == label].mean(0))
            centers2.append(inputs2[targets == label].mean(0))
        centers1 = torch.stack(centers1)
        centers2 = torch.stack(centers2)

        # Map targets to continuous indices
        mapped_targets = torch.tensor([label_map[label.item()] for label in targets], device=targets.device)

        # Distance to centers
        dist_pc1 = (inputs1 - centers1[mapped_targets]).pow(2).sum(1).sqrt().clamp(min=self.margin1)
        dist_pc2 = (inputs2 - centers2[mapped_targets]).pow(2).sum(1).sqrt().clamp(min=self.margin1)

        # Pairwise distance for centers1 and centers2
        dist1 = torch.pow(centers1, 2).sum(dim=1, keepdim=True).expand(len(label_uni), len(label_uni))
        dist1 = dist1 + dist1.t() - 2 * centers1.mm(centers1.t())
        dist1 = dist1.clamp(min=1e-12).sqrt()

        dist2 = torch.pow(centers2, 2).sum(dim=1, keepdim=True).expand(len(label_uni), len(label_uni))
        dist2 = dist2 + dist2.t() - 2 * centers2.mm(centers2.t())
        dist2 = dist2.clamp(min=1e-12).sqrt()

        # Hardest positive and negative for both modalities
        mask = torch.eye(len(label_uni), device=inputs1.device).bool()
        dist_ap, dist_an = [], []
        for i in range(len(label_uni)):
            dist_ap.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist1[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Cross-modal contrastive loss using average features
        cross_dist = (centers1 - centers2).pow(2).sum(1).sqrt()
        cross_loss = self.contrastive_loss(cross_dist, torch.zeros_like(cross_dist), torch.ones_like(cross_dist))

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        ranking_loss_value = self.ranking_loss(dist_an, dist_ap, y)

        loss = dist_pc1.mean() + dist_pc2.mean() + ranking_loss_value + cross_loss

        return loss, dist_pc1.mean(), dist_pc2.mean(), ranking_loss_value, cross_loss

if __name__ == "__main__":
    imgs = torch.randn(80, 2048).cuda()
    cam_ids = torch.tensor([3] * 40 + [2] * 40).cuda()
    labels = torch.tensor([338, 338, 338, 338, 115, 115, 115, 115, 241, 241, 241, 241, 291, 291,
                           291, 291, 58, 58, 58, 58, 10, 10, 10, 10, 110, 110, 110, 110,
                           335, 335, 335, 335, 234, 234, 234, 234, 332, 332, 332, 332] * 2).cuda()

    loss = ImprovedCSLoss(10).cuda()

    loss(imgs, labels, cam_ids)