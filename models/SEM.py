import torch.nn as nn
import torch

class GradientComputation(nn.Module):
    def __init__(self):
        super(GradientComputation, self).__init__()
        self.conv_layers = {}
        self.relu = nn.ReLU()

    def get_masked_conv(self, channels):
        if channels not in self.conv_layers:
            mask_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding=2, groups=channels, bias=False)
            mask = 1 * torch.ones(1, 1, 5, 5)
            mask[0, 0, 2, 2] = -24 # 要将所有求和后-最中间24次，所以是1-25次
            mask_conv.weight.data = mask.repeat(channels, 1, 1, 1)
            for param in mask_conv.parameters():
                param.requires_grad = False
            self.conv_layers[channels] = mask_conv

        return self.conv_layers[channels]

    def forward(self, img):
        b, c, h, w = img.shape
        mask_conv = self.get_masked_conv(c)
        mask_conv.cuda()
        masked_output = mask_conv(img)
        masked_output = self.relu(masked_output)

        sum_output = masked_output / 24.0 # 做平均

        x = 0.9 * img + 0.25 * sum_output # 然后按照该式子进行加权就可以了

        return x

if __name__ == "__main__":
    model = GradientComputation().cuda()
    img = torch.randn(123,256,36,18).cuda()
    print(model(img).shape)