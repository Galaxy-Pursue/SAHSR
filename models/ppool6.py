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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        # Self-Attention 层
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [p, b, c] (p = sequence length, b = batch size, c = embedding dimension)

        # Self-Attention 部分
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # Residual connection + layer norm

        # Feed Forward 部分
        ff_output = self.ffn(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)  # Residual connection + layer norm

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()

        # Masked Self-Attention 层 (避免看到未来的信息)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [p, b, c] (p = sequence length, b = batch size, c = embedding dimension)
        b, p, c = x.shape
        tgt_mask = torch.triu(torch.ones(p, p), diagonal=1).bool().cuda()  # 上三角矩阵，用于遮蔽未来的位置
        # Masked Self-Attention 部分 (注意这里是 causal masking)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # Residual connection + layer norm

        # Feed Forward 部分
        ff_output = self.ffn(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)  # Residual connection + layer norm

        return x

# 创建一个 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, fn=None, moda_type='increase', rsa_model='bilstm'):
        super(BiLSTM, self).__init__()
        self.rsa_model = rsa_model
        print(f'rsa_model:{rsa_model}\n')
        assert rsa_model in ['bilstm', 'lstm', 'transformerEncoder', 'transformerDecoder']
        if rsa_model == 'bilstm':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 将双向LSTM的输出变为C维度
        elif rsa_model == 'lstm':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)
            self.fc = nn.Linear(hidden_size, hidden_size)  # 将双向LSTM的输出变为C维度
        elif rsa_model == 'transformerEncoder':
            self.transformer = TransformerEncoderBlock(input_size, 4, hidden_size)
        elif rsa_model == 'transformerDecoder':
            self.transformer = TransformerDecoderBlock(input_size, 4, hidden_size)
        else:
            raise NotImplementedError

        self.fn = fn
        self.moda_type = moda_type

    def forward(self, x, sub, labels):
        if (self.rsa_model == 'bilstm') or (self.rsa_model == 'lstm'):
            h_lstm, _ = self.lstm(x)  # LSTM 输出，h_lstm 的大小是 (batch, N, hidden_size*2)
            T = h_lstm.size(1)  # 序列长度
            loss_moda = 0
            if self.training:
                if self.fn is not None:
                    for i in range(T):
                        # loss_cs, pc, ori_an = self.fn(h_lstm[:, i, :].float(), labels)
                        # loss_pc += pc
                        # if i < 3: continue
                        # bg_loss_kl, _ = self.fn(h_lstm[:, i, :].float(), labels, sub)
                        kl_loss_12, bg_loss_kl, cross_kl_loss_12, cross_kl_loss_21 = self.fn(h_lstm[:, i, :].float(),
                                                                                             labels, sub)
                        # loss_moda += kl_loss_12 *((T-1-i)/(T-1)) + cross_kl_loss_12 * ((i)/(T-1))
                        # loss_moda += kl_loss_12 * ((T - 1 - i) / (T - 1))
                        if self.moda_type == 'increase':
                            loss_moda += cross_kl_loss_12 * ((i) / (T - 1))
                        elif self.moda_type == 'decrease':
                            loss_moda += cross_kl_loss_12 * ((T-1-i) / (T - 1))
                        else:
                            loss_moda += cross_kl_loss_12 * (1/2)
                        # loss_moda += cross_kl_loss_12
            out = h_lstm[:, -1, :]  # 取最后一个时间步的输出，大小是 (batch, hidden_size*2)

            out = self.fc(out)  # 经过全连接层，大小是 (batch, hidden_size)
            return out, loss_moda

        else:
            out = self.transformer(x)
            out = torch.mean(out, dim=1)

            return out, torch.zeros(()).cuda()


class PyramidPooling6(nn.Module):
    def __init__(self, h_split=3, w_split=2, channel=2048, mode='avg', learnable=False, p=1.0, fn=None, moda_type='increase', rsa_model='bilstm'):
        super(PyramidPooling6, self).__init__()
        self.h_split = h_split
        self.w_split = w_split
        self.channel = channel
        self.mode = mode
        self.moda_type = moda_type
        print(f'moda_type:{self.moda_type}')

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

        self.rnn_conclu = BiLSTM(input_size=channel, hidden_size=channel, num_layers=1, fn=fn, moda_type=self.moda_type, rsa_model=rsa_model)

    def forward(self, x, sub, labels):
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
        x_conclu, loss_tgsa = self.rnn_conclu(x_patch, sub, labels)
        res = torch.cat((x_global.squeeze(1), x_conclu), dim=1)
        return res, loss_tgsa, x_patch.reshape(b, -1)



if __name__ == "__main__":
    x = torch.randn(123, 2048, 18, 9)

    pp = PyramidPooling5(h_split=3, w_split=2, mode='oth', learnable=True)

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
