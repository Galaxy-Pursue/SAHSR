import torch
from torch import nn 

class IALoss(nn.Module):
    def __init__(self,person_num,alpha = 0):
        super(IALoss, self).__init__()
        self.sum_pid = person_num
        self.alpha = alpha
    def cal_center(self,x,labels):
        uni_labels = torch.unique(labels)
        Cs = []
        for i, uni_label in enumerate(uni_labels):
            mask = torch.where(labels == uni_label,torch.tensor(1),torch.tensor(0)).bool()
            Cs.append(x[mask,:,:,:].mean(dim=0))
        return Cs
            
            
        
    def forward(self,x,labels):
        modality_size = x.shape[0]//3
        x_inf = x[:modality_size,:,:,:]
        labels_inf = labels[:modality_size]
        x_vis = x[modality_size:2*modality_size,:,:,:]
        labels_vis = labels[modality_size:2*modality_size]
        x_aux = x[2*modality_size:,:,:,:]
        labels_aux = labels[2*modality_size:]
        c_inf = self.cal_center(x_inf, labels_inf)
        c_vis = self.cal_center(x_vis, labels_vis)
        c_aux = self.cal_center(x_aux, labels_aux)
        uni_labels = torch.unique(labels)
        for i in range(len(uni_labels)):
            n_inf = torch.norm(c_aux[i]-c_inf[i])
            n_vis = torch.norm(c_aux[i]-c_vis[i])
            n_max = max(n_inf,n_vis)
            n_min = []
            for j in range(len(uni_labels)):
                if i == j: continue
                n_min.append(torch.norm(c_aux[i]-c_inf[j]))
                n_min.append(torch.norm(c_aux[i]-c_vis[j]))
            n_min = min(n_min)
            
            if i == 0:
                ialoss = n_max-n_min+self.alpha
            else:
                ialoss += (n_max-n_min+self.alpha)
        
        if ialoss<0: ialoss *= 0
        return ialoss
                
            
        
def init():
    gen = []
    for i in range(10):
        for _ in range(4):
            gen.append(i)
    return gen

if __name__ == '__main__':
    labels = init()+init()+init()
    labels = torch.tensor(labels)
    input = torch.randn(120, 2048, 18, 9)
    module = IALoss(10)
    print(module(input,labels))