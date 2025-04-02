import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# import matplotlib.pyplot as plt
import numpy as np
# torch.autograd.set_detect_anomaly(True)


def dist(feature1, feature2, type='Euclid'):
    """
    注意inforce的思想是让和同类的相似度尽可能大，不是让和同类的距离尽可能的大！！！
    """
    n = feature1.size()[0]
    assert type in ['Euclid', 'cos_dist']
    if type == 'Euclid':
        xx = torch.pow(feature1, 2).sum(dim=-1, keepdim=True).repeat(1, n)
        yy = torch.pow(feature2, 2).sum(dim=-1, keepdim=True).repeat(1, n).T
        xy = feature1 @ feature2.T
        dist = (xx + yy - 2 * xy).clamp(min=1e-12).sqrt()
        return dist
    else :
        cos = torch.matmul(feature1, feature2.transpose(-1, -2))
        xx = torch.pow(feature1, 2).sqrt().sum(dim=-1, keepdim=True).repeat(1, n)
        yy = torch.pow(feature2, 2).sqrt().sum(dim=-1, keepdim=True).repeat(1, n).T
        cos_sim = cos / (xx * yy)
        cos_dist = 1 - cos_sim
        return cos_dist


def sim(feature1, feature2, type='cos', eps=1e-6):
    n = feature1.size()[0]
    assert type in ['cos', 'cos_sim']
    cos = torch.matmul(feature1, feature2.transpose(-1, -2))
    if type == 'cos':
        return cos
    else:
        xx = torch.pow(feature1, 2).sqrt().sum(dim=-1, keepdim=True).repeat(1, n)
        yy = torch.pow(feature2, 2).sqrt().sum(dim=-1, keepdim=True).repeat(1, n).T
        cos_sim = cos / (xx * yy + eps)
        return cos_sim

class InfoNce(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    使同一类别的样本之间的相似度尽可能大，不同类别的样本之间的相似度尽可能小
    """

    def __init__(self,
                 temperature=0.07,
                 num_instance=4):

        super(InfoNce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance

    def forward(self, features):
        """
        :param features: (B, C, T)
        :param labels: (B)
        :return:
        """
        b, c, t = features.shape # 记录batchsz，channel，time(eg: 32*2048*4)

        ni = self.ni
        # 因为batch中连续的num_instance的ID相同，因此这里reshape一下
        features = features.reshape(b//ni, ni, c, t).permute(0, 3, 1, 2).reshape(b//ni, t*ni, c) # 8*16*2048（16是4个轨迹*每个轨迹4帧）
        # fetures变换完后，每一个batch指一个人，每一个batch中，依次排列为第一段视频的所有帧，第二段视频的所有帧。。。
        # reshape的机理好像是按照索引从小到大按照想要得到的维度往里面填数字？？？
        features = F.normalize(features, dim=-1)
        labels = torch.arange(0, t).reshape(t, 1).repeat(1, ni).reshape(t*ni, 1)
        # (t*ni, t*ni)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (t*ni, t*ni)，如果两幅图片来自同一个视频序列则为1，反之为0
        mask_pos = (1 - torch.eye(t*ni)).cuda()
        mask_pos = (mask * mask_pos).unsqueeze(0)  # mask_pos的第i行：与ilabel相等的除了自己的

        # 计算余弦相似度
        # 当输入有多维时，把多出的一维作为batch提出来，其他部分做矩阵乘法。
        # cos的第i行：i与所有别的样本的相似度
        cos = torch.matmul(features, features.transpose(-1, -2))

        logits = torch.div(cos, self.temperature)  # logits: (b//ni, t*ni, t*ni) (8,16,16)
        exp_neg_logits = (logits.exp() * (1-mask)).sum(dim=-1, keepdim=True)  #计算不同instance相似度的和

        # 下面一行和原论文给出的公式在分母上略有不同，原论文是所有与i不同的例子的和，包括正例；这里是负例+自己
        log_prob = logits - torch.log(exp_neg_logits + logits.exp())
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())
        loss = - loss
        return loss


class original_inforce(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    使同一类别的样本之间的相似度尽可能大，不同类别的样本之间的相似度尽可能小
    """

    def __init__(self,
                 temperature=0.07,
                 num_instance=4,
                 inforce_sim_type='cos',
                 ):

        super(original_inforce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance
        self.inforce_sim_type = inforce_sim_type
        print(f'use sim_type:{self.inforce_sim_type}')

    def forward(self, features, labels=None):
        """
        :param features: (B, C)
        :param labels: (B)
        :return:
        """
        b, c = features.shape  # 记录batchsz，channel

        k = self.ni
        p = b // k

        features = F.normalize(features, dim=-1)
        if labels == None:
            labels = torch.arange(0, p).reshape(p, 1).repeat(1, k).reshape(p*k, 1)
        # (p*k, p*k)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (p*k, p*k)，如果两幅图片来自同一个ren则为1，反之为0
        mask_pos = (1 - torch.eye(p*k)).cuda()
        mask_pos = (mask * mask_pos)  # mask_pos的第i行：与i label相等的除了自己的

        # 计算余弦相似度
        # cos的第i行：i与所有别的样本的相似度
        if self.inforce_sim_type == 'cos':
            cos = torch.matmul(features, features.transpose(-1, -2))  # (p*k, p*k)
        else:
            cos = sim(features, features, type='cos_sim')
        logits = torch.div(cos, self.temperature)  # logits: (p*k, p*k)
        exp_neg_logits = (logits.exp() * (1-mask)).sum(dim=-1, keepdim=True)  # 计算不同label相似度的exp的和

        # 下面一行和原论文给出的公式在分母上略有不同，原论文是所有与i不同的例子的和，包括正例；这里是负例+与i同label的（除了i）
        log_prob = logits - torch.log(exp_neg_logits + logits.exp())  # 括号中的i，j表示：与i不同label相似度的exp的和 + i与j相似度exp
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())  # 将上一行的j换成与i同label的instance（除了i）
        loss = - loss
        return loss


class modality_inforce(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 num_instance=4,
                 inforce_sim_type='cos',
                 ):
        super(modality_inforce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance
        self.inforce_sim_type = inforce_sim_type

    def forward(self, a_feature, b_feature, a_label, b_label):
        b = a_feature.size()[0]  # 实际上是p*k//2
        mask = torch.eq(a_label.view(-1, 1), b_label.view(1, -1)).float().cuda()  # (p*k, p*k)，如果两幅图片来自同一个人则为1，反之为0
        mask_pos = (1 - torch.eye(b)).cuda()
        mask_pos = (mask * mask_pos)  # mask_pos的第i行：与i label相等的除了自己的

        # 计算余弦相似度
        # cos的第i行：i与所有别的样本的相似度
        if self.inforce_sim_type == 'cos':
            cos = torch.matmul(a_feature, b_feature.transpose(-1, -2))  # (p*k//2, p*k//2)
        else:
            cos = sim(a_feature, b_feature, type='cos_sim')
        logits = torch.div(cos, self.temperature)  # logits: (p*k//2, p*k//2)
        exp_neg_logits = (logits.exp() * (1 - mask)).sum(dim=-1, keepdim=True)  # 计算不同label相似度的exp的和

        # 下面一行和原论文给出的公式在分母上略有不同，原论文是所有与i不同的例子的和，包括正例；这里是负例+与i同label的（除了i）
        log_prob = logits - torch.log(exp_neg_logits + logits.exp())  # 括号中的i，j表示：与i不同label相似度的exp的和 + i与j相似度exp
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())  # 将上一行的j换成与i同label的instance（除了i）
        loss = - loss
        return loss


class vi_inforce(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 num_instance=4,
                 intra=True,
                 inter=True,
                 sigma=1,
                 inforce_sim_type = 'cos',
                 ):
        super(vi_inforce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance
        self.intra=intra
        self.inter=inter
        self.sigma=sigma
        self.inforce_sim_type=inforce_sim_type
        print(f'use sim_type:{self.inforce_sim_type}')
        if self.intra:
            self.intra_R2R_fn = modality_inforce(self.temperature, self.ni, self.inforce_sim_type)
            self.intra_V2V_fn = modality_inforce(self.temperature, self.ni, self.inforce_sim_type)
        if self.inter:
            self.inter_R2V_fn = modality_inforce(self.temperature, self.ni, self.inforce_sim_type)
            self.inter_V2R_fn = modality_inforce(self.temperature, self.ni, self.inforce_sim_type)

    def forward(self, features, labels=None):
        """
        :param features: (B, C)
        :param labels: (B)
        :return:
        """
        b, c = features.shape  # 记录batchsz，channel

        k = self.ni
        p = b // k

        features = F.normalize(features, dim=-1)
        if labels == None:
            labels = torch.arange(0, p).reshape(p, 1).repeat(1, k).reshape(p * k, 1)

        feature_R = features[:b//2]
        feature_V = features[b//2:]
        label_R = labels[:b//2]
        label_V = labels[b//2:]

        loss_intra = 0
        loss_inter = 0
        if self.intra:
            loss_intra = self.intra_R2R_fn(feature_R, feature_R, label_R, label_R) + self.intra_V2V_fn(feature_V, feature_V, label_V, label_V)
        if self.inter:
            loss_inter = self.inter_R2V_fn(feature_R, feature_V, label_R, label_V) + self.inter_V2R_fn(feature_V, feature_R, label_V, label_R)

        loss = loss_intra + self.sigma * loss_inter

        return loss


class centerNCE(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 num_instance=4):
        super(centerNCE, self).__init__()
        self.temperature = temperature
        self.ni = num_instance

    def forward(self, features, labels):
        n = features.size(0)
        # 这个normalized是我新加的，之前使用0.07的温度系数exp后直接炸了，这次加上normalized再用0.07的系数
        features = F.normalize(features, dim=-1)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(features[labels == labels[i]].mean(0))
        centers = torch.stack(centers)

        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (p*k, p*k)，如果两幅图片来自同一个视频序列则为1，反之为0
        mask_pos = torch.eye(n).cuda()

        # 计算余弦相似度
        # cos的第i行：i与所有别的样本的相似度
        cos = torch.matmul(features, centers.transpose(-1, -2))  # (p*k, p*k)
        # cos = sim(features, centers, type='cos_sim')

        logits = torch.div(cos, self.temperature)  # logits: (p*k, p*k)
        exp_neg_logits = (logits.exp() * (1 - mask)).sum(dim=-1, keepdim=True) / self.ni  # 计算不同label相似度的exp的和

        # 下面一行和原论文给出的公式在分母上略有不同，原论文是所有与i不同的例子的和，包括正例；这里是负例+与i同label的（除了i）
        log_prob = logits - torch.log(exp_neg_logits + logits.exp())  # 括号中的i，j表示：与i不同label相似度的exp的和 + i与j相似度exp
        centerNCE = (log_prob * mask_pos).sum() / (mask_pos.sum())  # 将上一行的j换成与i同label的instance（除了i）
        centerNCE = - centerNCE

        return centerNCE


class margin_centerNCE(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 num_instance=4,
                 up_margin=1,
                 down_margin=-1,
                 ):
        super(margin_centerNCE, self).__init__()
        self.temperature = temperature
        self.ni = num_instance
        # margin取1和-1时与centerNCE一样，通过控制margin来控制过拟合的程度
        self.up_margin = up_margin
        self.down_margin = down_margin


    def forward(self, features, labels):
        n = features.size(0)
        # 这个normalized是我新加的，之前使用0.07的温度系数exp后直接炸了，这次加上normalized再用0.07的系数
        features = F.normalize(features, dim=-1)  # 使用沿着dim=-1的normalize可能有点不妥

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(features[labels == labels[i]].mean(0))
        centers = torch.stack(centers)

        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (p*k, p*k)，如果两幅图片来自同一个视频序列则为1，反之为0
        mask_pos = torch.eye(n).cuda()

        # 计算余弦相似度
        # cos的第i行：i与所有别的样本的相似度
        cos = torch.matmul(features, centers.transpose(-1, -2))  # (p*k, p*k)
        cos_up = cos.clamp(max=self.up_margin)
        cos_down = cos.clamp(min=self.down_margin)
        logits_up = torch.div(cos_up, self.temperature)
        logits_down = torch.div(cos_down, self.temperature)

        # logits = torch.div(cos, self.temperature)  # logits: (p*k, p*k)
        exp_neg_logits = (logits_down.exp() * (1 - mask)).sum(dim=-1, keepdim=True) / self.ni  # 计算不同label相似度的exp的和

        # 下面一行和原论文给出的公式在分母上略有不同，原论文是所有与i不同的例子的和，包括正例；这里是负例+与i同label的（除了i）
        log_prob = logits_up - torch.log(exp_neg_logits + logits_up.exp())  # 括号中的i，j表示：与i不同label相似度的exp的和 + i与j相似度exp
        centerNCE = (log_prob * mask_pos).sum() / (mask_pos.sum())  # 将上一行的j换成与i同label的instance（除了i）
        centerNCE = - centerNCE

        return centerNCE

# def plot_(features, labels):
#     data_np = features.detach().cpu().numpy()
#     labels_np = labels.detach().cpu().numpy()
#
#     # 获取所有的标签值
#     unique_labels = np.unique(labels_np)
#
#     # 为每个标签分配一个颜色
#     colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'yellow', 'gray']  # 可以根据需要添加更多颜色
#
#     # 绘制散点图
#     for i, label in enumerate(unique_labels):
#         plt.scatter(data_np[labels_np == label, 0], data_np[labels_np == label, 1], color=colors[i],
#                     label=f'label {label}')
#
#     # 添加图例
#     plt.legend()
#
#     # 显示图像
#     plt.show()

# if __name__ == '__main__':
#     seed = 0
#     np.random.RandomState(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     p, k = 8, 8
#     c = 8
#     features = torch.rand(p*k, c).cuda()
#     labels = torch.arange(0, p).reshape(p, 1).repeat(1, k//2).reshape(p * k//2, 1).repeat(2, 1).squeeze().cuda()
#     model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).cuda()
#     lr=1e-4
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = margin_centerNCE(num_instance=k)
#
#     original = model(features)
#     plot_(original, labels)
#
#     for i in range(5000):
#         optimizer.zero_grad()
#         output = model(features)
#         loss = loss_fn(output, labels)
#         loss.backward()
#         optimizer.step()
#         if i % 200 == 0:
#             plot_(output, labels)
#             pass
#
#
#
#
#
