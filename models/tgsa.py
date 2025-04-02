import torch 
from torch import nn 
import torch.nn.functional as F 


def pairwise_dist(x, y):
    # Compute pairwise distance of vectors 计算两两向量之间的欧式距离
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist

def kl_soft_dist(feat1,feat2):
    n_st = feat1.size(0)
    dist_st = pairwise_dist(feat1, feat2)
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)
    for i in range(n_st):  # 将同一类样本中自己与自己的距离舍弃
        mask_st_1[i, i] = 0
    dist_st_2 = []
    for i in range(n_st):
        dist_st_2.append(dist_st[i][mask_st_1[i]])  # modality_a的第i个元素与除了自己以为所有modality_b的距离保存为一个向量
    dist_st_2 = torch.stack(dist_st_2)
    return dist_st_2


def Bg_kl(logits1, logits2):  # 输入:(60,206),(60,206) 这和他论文里面写的不一样，他用的距离矩阵算的kl散度
    KL = nn.KLDivLoss(reduction='batchmean')
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21
    return kl_loss_12, bg_loss_kl

def cal_tgsa(feat, sub):
    # sf_sp_dist_v = kl_soft_dist(sp_pl[sub == 0], sp_pl[sub == 0])
    # sf_sp_dist_i = kl_soft_dist(sp_pl[sub == 1], sp_pl[sub == 1])
    sf_sh_dist_v = kl_soft_dist(feat[sub == 0], feat[sub == 0])
    sf_sh_dist_i = kl_soft_dist(feat[sub == 1], feat[sub == 1])
    # half_B0 = feat[sub == 0].shape[0] // 2
    # feat_half0 = feat[sub == 0][:half_B0]  # 拿出一半的可见光特征
    # half_B1 = feat[sub == 1].shape[0] // 2
    # feat_half1 = feat[sub == 1][:half_B1]  # 拿出一半的红外线特征
    # feat_cross = torch.cat((feat_half0, feat_half1), dim=0)
    sf_sh_dist_vi = kl_soft_dist(feat[sub == 0], feat[sub == 1])



    # _, kl_inter_v = Bg_kl(sf_sh_dist_v, sf_sp_dist_v)
    # _, kl_inter_i = Bg_kl(sf_sh_dist_i, sf_sp_dist_i)


    _, kl_intra1 = Bg_kl(sf_sh_dist_v, sf_sh_dist_i)
    _, kl_intra2 = Bg_kl(sf_sh_dist_v, sf_sh_dist_vi)  # 这个也是论文里面没有的
    _, kl_intra3 = Bg_kl(sf_sh_dist_vi, sf_sh_dist_i)  # 这个也是论文里面没有的

    kl_intra = kl_intra1 + kl_intra2 + kl_intra3

    return kl_intra



if __name__ == "__main__":
    inp = torch.ones(4,5);
    dist = kl_soft_dist(inp, inp)
    