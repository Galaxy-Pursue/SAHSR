import torch
import torch.nn as nn
from torch.nn import functional as F

from models.ModalityAlignmentLoss import ModaReduceLoss
from models.rsa import RecurrentSemanticAggregration
from models.SEM import GradientComputation
from models.extra_train_loss import group_loss
from models.resnet import resnet50, resnet18
from models.layers import CSLoss
from models.layers import DualBNNeck


class embed_net(nn.Module):
    def __init__(self, backbone, pretrained, drop_last_stride, modality_attention, **kwargs):
        super(embed_net, self).__init__()

        self.ban_ppool = kwargs.get('ban_ppool', False)
        self.SEM = GradientComputation()
        
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            self.D = 2048
        elif backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            self.D = 512


            
    def forward(self, inputs, cam_ids, mode=None, **kwargs):

        if self.training:
            global_feat = self.backbone.conv1(inputs)
            global_feat = self.backbone.bn1(global_feat)
            global_feat = self.backbone.relu(global_feat)
            global_feat = self.backbone.maxpool(global_feat)

            global_feat = self.backbone.layer1(global_feat)

            global_feat = self.SEM(global_feat)

            global_feat = self.backbone.layer2(global_feat)

            global_feat = self.SEM(global_feat)
            global_feat = self.backbone.layer3(global_feat)

            if self.backbone.modality_attention > 1:
                    global_feat = self.backbone.MAM3(global_feat)

            global_feat = self.backbone.layer4(global_feat)

            
            if self.backbone.modality_attention > 0:
                global_feat = self.backbone.MAM4(global_feat)
        else:
            if mode == 'inf':
                global_feat = self.backbone.conv1(inputs)
                global_feat = self.backbone.bn1(global_feat)
                global_feat = self.backbone.relu(global_feat)
                global_feat = self.backbone.maxpool(global_feat)
            else: 
                global_feat = self.backbone.conv1(inputs)
                global_feat = self.backbone.bn1(global_feat)
                global_feat = self.backbone.relu(global_feat)
                global_feat = self.backbone.maxpool(global_feat)

            global_feat = self.backbone.layer1(global_feat)
            global_feat = self.SEM(global_feat)

            global_feat = self.backbone.layer2(global_feat)
            global_feat = self.SEM(global_feat)
            global_feat = self.backbone.layer3(global_feat)

            
            if self.backbone.modality_attention > 1:
                    global_feat = self.backbone.MAM3(global_feat)
            global_feat = self.backbone.layer4(global_feat)
            
            if self.backbone.modality_attention > 0:
                global_feat = self.backbone.MAM4(global_feat)
        return global_feat

class Baseline(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()
        
        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        self.moda_type = kwargs.get('moda_type', 'increase')
        self.rsa_model = kwargs.get('rsa_model', 'bilstm')

        self.p_size = kwargs.get('p_size', 12)
        self.k_size = kwargs.get('k_size', 8)
        self.top_k = kwargs.get('top_k', 4)
        self.extra_img_num = kwargs.get('extra_img_num', 2)
        self.part_num = 1  # 11 # kwargs.get('num_parts', 7)
        self.ma_w = kwargs.get('ma_w', 0)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.group_cs_w = kwargs.get('group_cs_w', 0.0)
        print(f'group_cs_w:{self.group_cs_w}')
        self.group_ce_w = kwargs.get('group_ce_w', 0.0)
        print(f'group_ce_w:{self.group_ce_w}')
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        self.use_rsa = kwargs.get('use_rsa', 1)


        if self.use_rsa:
            print(f'use_rsa:{self.use_rsa}\n')

        
        self.backbone = embed_net(backbone, True, drop_last_stride, modality_attention)
        D = self.backbone.D 
        self.base_dim = D
        self.dim = D
        
        if not self.use_rsa:
            self.part_num = 6
        else:
            print('Intializing RSA!')
            self.moda_loss_fn = ModaReduceLoss()
            self.rsa = RecurrentSemanticAggregration(fn = self.moda_loss_fn, mode='avg', moda_type=self.moda_type, rsa_model=self.rsa_model)
        self.bn_neck = DualBNNeck(self.base_dim + self.dim * self.part_num)
        
        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
        
        self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')

        self.weight_KL = kwargs.get('weight_KL', 2.0)
        self.update_rate = kwargs.get('update_rate', 0.2)
        self.update_rate_ = self.update_rate
        
        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num , num_classes, bias=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2) # 0.01 # 0.7

        self.group_ce_fn = group_loss(self.top_k, self.extra_img_num, self.p_size, 'ce')
        # self.group_inforce_fn = group_loss(self.top_k, self.extra_img_num, self.p_size, 'inforce')
        self.group_cs_fn = group_loss(self.top_k, self.extra_img_num, self.p_size, 'cs', margin1=self.margin1, margin2=self.margin2)


        
    def forward(self, inputs, labels=None, mode=None, extra_train=False, **kwargs):
        cam_ids = kwargs.get("cam_ids")
        sub = (cam_ids == 3) + (cam_ids == 6)
        
        global_feat = self.backbone(inputs, cam_ids, mode)
        
        b, c, w, h = global_feat.shape

        if not self.use_rsa:
            part_feat = F.adaptive_avg_pool2d(global_feat, (3, 2)).view(b, -1)
            global_feat = global_feat.mean(dim=(2, 3))
            feats = torch.cat([global_feat, part_feat], dim=1)
            loss_ma = torch.tensor(0.0, requires_grad=False).cuda()
        else:
            feats, loss_ma, x_patch = self.rsa(global_feat, sub, labels)

        if not self.training:
            feats = self.bn_neck(feats, sub)
            return feats
        else:
            pass

            return self.train_forward(feats, labels, sub, loss_ma, extra_train, **kwargs)

    def train_forward(self, feat, labels, sub, loss_ma, extra_train=False, **kwargs):
        metric = {}

        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)

        if extra_train == True:
            loss_cs_group = self.group_cs_fn(feat.float(), labels)

        
        feat = self.bn_neck(feat, sub)
    
        logits = self.classifier(feat)

        loss_id = self.ce_loss_fn(logits.float(), labels)

        if extra_train == True:
            loss_ce_group = self.group_ce_fn(logits.float(), labels)

        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})
        
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        
        logits_v = self.visible_classifier(feat[sub == 0])
        loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])
        logits_i = self.infrared_classifier(feat[sub == 1])
        loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])
        
        logits_m = torch.cat([logits_v, logits_i], 0).float()
        with torch.no_grad():
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                + self.infrared_classifier.weight.data * self.update_rate
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                + self.visible_classifier.weight.data * self.update_rate

            logits_v_ = self.infrared_classifier_(feat[sub == 0])
            logits_i_ = self.visible_classifier_(feat[sub == 1])
            logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()

        loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1)) 

        metric.update({'id': loss_id.data})
        metric.update({'cs': loss_cs.data})
        metric.update({'ma': loss_ma.data})

        loss = loss_id + loss_cs * self.cs_w + loss_ma * self.ma_w

        if extra_train == False:
            pass
        else:
            loss = loss + self.group_ce_w * loss_ce_group + self.group_cs_w * loss_cs_group

        return loss, metric, logits
