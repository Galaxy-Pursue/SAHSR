import torch
from torch import nn
from torch.nn import functional as F

class ModaReduceLoss(nn.Module):
    def __init__(self):
        super(ModaReduceLoss, self).__init__()
        self.KL = nn.KLDivLoss(reduction='batchmean')
    
    def distance_matrix(self, x, y):
        dist_m = torch.cdist(x, y, p=2)
        return dist_m
    
    def forward(self, features, ids, sub):
        unique_ids = torch.unique(ids)
        total_kl_loss_12 = 0
        total_bg_loss_kl = 0
        num_ids = unique_ids.size(0)
        
        for person_id in unique_ids:
            mask = (ids == person_id)
            features_mod1 = features[mask & sub]
            features_mod2 = features[mask & ~sub]
            
            if features_mod1.size(0) > 1 and features_mod2.size(0) > 1:
                dist_m1 = self.distance_matrix(features_mod1, features_mod1)
                dist_m2 = self.distance_matrix(features_mod2, features_mod2)
                
                P = F.softmax(-dist_m1, dim=1)
                Q = F.softmax(-dist_m2, dim=1)
                
                kl_loss_12 = self.KL(P.log(), Q)
                kl_loss_21 = self.KL(Q.log(), P)
                
                bg_loss_kl = kl_loss_12 + kl_loss_21
                
                total_kl_loss_12 += kl_loss_12
                total_bg_loss_kl += bg_loss_kl
        
        avg_kl_loss_12 = total_kl_loss_12 / num_ids
        avg_bg_loss_kl = total_bg_loss_kl / num_ids
        
        return avg_kl_loss_12, avg_bg_loss_kl
