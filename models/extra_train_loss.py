import torch
from torch import nn
import torch.nn.functional as F

from models.model_base8.infoNCE import original_inforce, centerNCE
from models.model_base8.layers import CSLoss

class group_loss(nn.Module):
    def __init__(self, top_k, extra_img_num, p_size, loss_type, margin1=0.01, margin2=0.7):
        super(group_loss, self).__init__()
        self.top_k = top_k
        self.extra_img_num = extra_img_num
        self.img_num = self.top_k*self.extra_img_num
        self.p_size = p_size

        assert loss_type in ['ce', 'inforce', 'centerNCE', 'cs']
        self.loss_type = loss_type

        if loss_type == 'ce':
            # 可不可以不把top_k个logits单独拿出来而是使用nn.ce的weight参数，使得logits越大的weight越大
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'inforce':
            self.loss_fn = original_inforce(num_instance=self.extra_img_num)
        elif loss_type == 'centerNCE':
            self.loss_fn = centerNCE(num_instance=self.extra_img_num)
        elif loss_type == 'cs':
            self.loss_fn = CSLoss(k_size=self.extra_img_num, margin1=margin1, margin2=margin2)
        else:
            raise TypeError('not supported loss function')




    def forward(self, feats, labels):
        # 在extra_train中，batch一共被分为p组，每组top_k个人，每个人extra_img_num张图像
        # 前面半个batch是p组，每组top_k个人，每个人extra_img_num//2张ir图；
        # 后面半个batch是p组，每组top_k个人，每个人extra_img_num//2张rgb图；
        n = feats.size()[0]
        device = feats.device
        group_loss_value = torch.tensor(0.0, device=device, requires_grad=False)
        img_num = self.top_k*self.extra_img_num//2

        # 切分批次数据
        ir_feats = feats[:n // 2].reshape(self.p_size, img_num, -1)
        rgb_feats = feats[n // 2:].reshape(self.p_size, img_num, -1)
        ir_labels = labels[:n // 2].reshape(self.p_size, img_num)
        rgb_labels = labels[n // 2:].reshape(self.p_size, img_num)
        # feats:[p, img_num, c]  labels:[p, img_num]
        feats = torch.cat([ir_feats, rgb_feats], dim=1)
        labels = torch.cat([ir_labels, rgb_labels], dim=1)

        for person_id in range(self.p_size):
            group_feats = feats[person_id]
            group_labels = labels[person_id]

            # 找到 group_labels 中的 k 个不同类别
            unique_group_labels = torch.unique(group_labels)

            # 创建映射字典，将原始类别映射到 0~k-1 的范围内
            label_to_index = {lab.item(): idx for idx, lab in enumerate(unique_group_labels)}

            # 创建新的 labels 向量
            group_labels_new = torch.tensor([label_to_index[lab.item()] for lab in group_labels], dtype=torch.long, device=device)

            if self.loss_type == 'ce':
                # 通过高级索引提取 feats 中对应的元素
                indices = unique_group_labels.unsqueeze(0).expand(group_feats.size(0), -1)
                group_feats_new = torch.gather(group_feats, 1, indices)
            else:
                group_feats_new = group_feats


            # 计算交叉熵损失
            if self.loss_type == 'cs':
                group_loss_value += self.loss_fn(group_feats_new, group_labels_new)[0]
            else:
                group_loss_value += self.loss_fn(group_feats_new, group_labels_new)

        return group_loss_value