import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.cuda.amp as amp


def custom_relu(input_tensor):
    # CReLU(x) = min(max(0, x),1)
    return torch.minimum(torch.maximum(input_tensor, torch.tensor(0.0)), torch.tensor(1.0))


def ND_init(num_points, mean=0, stddev=1, scale=2):
    x_values = np.linspace(-scale * stddev, scale * stddev, num_points)
    y_values = np.exp(-0.5 * ((x_values - mean) / stddev) ** 2)
    return torch.from_numpy(y_values)


"""
找到共同的频域
有点像SAFL
"""


class AdaFilter(nn.Module):
    def __init__(self, h, w, c=3):
        super(AdaFilter, self).__init__()
        # 创建一个可学习的参数
        self.chann_param = nn.Parameter(torch.ones(c, 1, 1))
        self.horiz_param = nn.Parameter(torch.ones((1, w)))
        self.verti_param = nn.Parameter(torch.ones((h, 1)))

    def forward(self, x):
        with amp.autocast(enabled=False):
            x_amp, x_phase = self.image_to_spectrum(x)
            x_amp_low, x_amp_rev = self.pass_filter(x_amp)
            res = self.spectrum_to_image(x_amp_low, x_phase)
            res_rev = self.spectrum_to_image(x_amp_rev, x_phase)
        return res, res_rev 

    def image_to_spectrum(self, image_tensor):
        # 应用二维傅里叶变换
        fft_image = torch.fft.fft2(image_tensor.float())
        fft_image_shifted = torch.fft.fftshift(fft_image)

        # 计算幅度谱和相位谱
        magnitude_spectrum = torch.abs(fft_image_shifted)
        phase_spectrum = torch.angle(fft_image_shifted)

        magnitude_spectrum = torch.log(magnitude_spectrum + 1)

        return magnitude_spectrum, phase_spectrum

    def spectrum_to_image(self, magnitude_spectrum, phase_spectrum):
        magnitude_spectrum = torch.exp(magnitude_spectrum) - 1
        # 重建复数表示
        complex_spectrum = magnitude_spectrum * torch.exp(1j * phase_spectrum)

        # 逆中心化处理
        complex_spectrum_shifted = torch.fft.ifftshift(complex_spectrum)

        # 应用逆傅里叶变换
        ifft_image = torch.fft.ifft2(complex_spectrum_shifted)

        # 获取实数部分
        reconstructed_image = ifft_image.real

        return reconstructed_image

    def pass_filter(self, ori_amp):
        # global_attn = custom_relu(self.verti_param) @ custom_relu(self.horiz_param) # H*W
        global_attn = F.relu(self.chann_param * ((self.verti_param @ self.horiz_param).unsqueeze(0)))
        global_attn = torch.clamp(global_attn, min=0.0, max=2.0)
        # global_attn = global_attn.unsqueeze(0) # 1*H*W
        global_attn_rev = torch.ones_like(global_attn)*2 - global_attn

        filtered_amp = ori_amp * global_attn
        filtered_amp_rev = ori_amp * global_attn_rev

        return filtered_amp, filtered_amp_rev


if __name__ == "__main__":
    input = torch.randn(80, 3, 288, 144)
    module = AdaFilter(h=288, w=144)
    print(module(input).shape)
