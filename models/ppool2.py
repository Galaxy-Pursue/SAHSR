import torch 
from torch import nn 
import torch.nn.functional as F 

class PyramidPooling2(nn.Module):
    def __init__(self, h_split = 3,w_split = 2, mode = 'avg', learnable=True):
        super(PyramidPooling2, self).__init__()
        self.h_split = h_split
        self.w_split = w_split
        if mode == 'max':
            self.height_pool = nn.AdaptiveMaxPool2d((h_split, 1))
            self.width_pool = nn.AdaptiveMaxPool2d((1, w_split))
            self.patch_pool = nn.AdaptiveMaxPool2d((h_split, w_split))
        else:
            self.height_pool = nn.AdaptiveAvgPool2d((h_split, 1))
            self.width_pool = nn.AdaptiveAvgPool2d((1, w_split))
            self.patch_pool = nn.AdaptiveAvgPool2d((h_split, w_split))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.learnable = learnable
        if learnable: self.param = nn.Parameter(torch.ones((3)))
        
    def forward(self, x):
        b, c, _, _ = x.shape
        x_height = self.height_pool(x).view(b, -1)
        x_width = self.width_pool(x).view(b, -1)
        x_patch = self.patch_pool(x).view(b, -1)
        x_global = self.global_pool(x).view(b, -1)
        # param = F.softmax(self.param, dim=0) * 3
        if self.learnable:
            param = F.relu(self.param)
            res = torch.cat((x_global,x_height*param[0],x_width*param[1],x_patch*param[2]),dim = 1)
        else:
            res = torch.cat((x_global,x_height,x_width,x_patch),dim = 1)
        return res
if __name__ == "__main__":
    b = 80
    global_feat = torch.randn(80, 2048, 18, 9)

    part_feat = F.adaptive_avg_pool2d(global_feat, (3, 3)).view(b, -1)
    print(part_feat.shape)
    module = PyramidPooling2(h_split=3, w_split=2)
    print(module(global_feat).shape)