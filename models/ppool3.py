import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGeM2d(nn.Module):
    def __init__(self, output_size, p=1.0):
        """
        初始化 GeM 池化层
        :param output_size: 输出的大小 (height, width)
        :param p: 初始化的 p 值
        :param eps: 防止数值不稳定的小值
        """
        super(AdaptiveGeM2d, self).__init__()
        self.output_size = output_size
        # self.p = nn.Parameter(torch.zeros(1) + p - 1)

    def forward(self, x, p):
        """
        前向传播
        :param x: 输入张量 (batch_size, channels, height, width)
        :return: 池化后的张量 (batch_size, channels, output_height, output_width)
        """
        # p = F.relu(self.p) + 1
        x = torch.pow(x, p)
        avg_layer = nn.AdaptiveAvgPool2d(output_size=self.output_size)
        # 自适应池化操作
        x = avg_layer(x)
        # 还原 GeM 操作的结果
        x = torch.pow(x, 1.0 / p)
        return x
    

# 创建一个 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 将双向LSTM的输出变为C维度

    def forward(self, x):
        h_lstm, _ = self.lstm(x)  # LSTM 输出，h_lstm 的大小是 (batch, N, hidden_size*2)
        out = h_lstm[:, -1, :]  # 取最后一个时间步的输出，大小是 (batch, hidden_size*2)
        out = self.fc(out)  # 经过全连接层，大小是 (batch, hidden_size)
        return out


class PyramidPooling3(nn.Module):
    def __init__(self, h_split=3, w_split=2, channel=2048, mode='avg', learnable=False, p = 1.0):
        super(PyramidPooling3, self).__init__()
        self.h_split = h_split
        self.w_split = w_split
        self.channel = channel
        self.mode = mode
        if mode == 'max':
            self.height_pool = nn.AdaptiveMaxPool2d((h_split, 1))
            self.width_pool = nn.AdaptiveMaxPool2d((1, w_split))
            self.patch_pool = nn.AdaptiveMaxPool2d((h_split, w_split))
        elif mode == 'avg':
            self.height_pool = nn.AdaptiveAvgPool2d((h_split, 1))
            self.width_pool = nn.AdaptiveAvgPool2d((1, w_split))
            self.patch_pool = nn.AdaptiveAvgPool2d((h_split, w_split))
        else:
            self.p = nn.Parameter(torch.zeros(1) + p - 1)
            self.height_pool = AdaptiveGeM2d((h_split, 1))
            self.width_pool = AdaptiveGeM2d((1, w_split))
            self.patch_pool = AdaptiveGeM2d((h_split, w_split))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.learnable = learnable
        if learnable: self.param = nn.Parameter(torch.ones((3)))

        self.rnn_conclu = BiLSTM(input_size=channel, hidden_size=channel,num_layers=1)


    def forward(self, x):
        b, c, _, _ = x.shape
        if self.mode == 'max' or self.mode == 'avg':
            x_height = self.height_pool(x).view(b, c, -1).transpose(1, 2)
            x_width = self.width_pool(x).view(b, c, -1).transpose(1, 2)
            x_patch = self.patch_pool(x).view(b, c, -1).transpose(1, 2)
            x_global = self.global_pool(x).view(b, c, -1).transpose(1, 2)
        else:
            p = F.relu(self.p) + 1
            x_height = self.height_pool(x, p).view(b, c, -1).transpose(1, 2)
            x_width = self.width_pool(x, p).view(b, c, -1).transpose(1, 2)
            x_patch = self.patch_pool(x, p).view(b, c, -1).transpose(1, 2)
            x_global = self.global_pool(x).view(b, c, -1).transpose(1, 2)
        # x_now = torch.cat((x_global, x_height, x_width, x_patch), dim=1)
        res = torch.cat((x_global.squeeze(1), self.rnn_conclu(x_patch)), dim=1)
        return res
        # param = F.softmax(self.param, dim=0) * 3
        # if self.learnable:
        #     param = F.relu(self.param)
        #     res = torch.cat((x_global, x_height * param[0], x_width * param[1], x_patch * param[2]), dim=1)
        # else:
        #     res = torch.cat((x_global, x_height, x_width, x_patch), dim=1)
        # return res


if __name__ == "__main__":

    x = torch.randn(123,2048,18,9)

    pp = PyramidPooling3(h_split=3, w_split=2, mode='oth', learnable=True)

    y = pp(x)
    print(y.shape)
    print("=================================")

    # 假设 N 和 C 的值
    N = 10  # sequence length
    C = 512  # feature dimension
    # 输入 tensor，大小为 (batch_size, N, C)
    batch_size = 1  # 这里假设 batch_size 为 1
    input_tensor = torch.randn(batch_size, N, C)

    # 定义 BiLSTM 模型
    hidden_size = C // 2  # 因为是双向LSTM，hidden_size 应该是 C 的一半
    model = BiLSTM(C, hidden_size)

    # 前向传播
    output_tensor = model(input_tensor)

    # 输出 tensor 的大小为 (batch_size, C)
    print(output_tensor.shape)  # 应该输出 torch.Size([1, 512])
