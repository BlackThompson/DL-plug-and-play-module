# LKA
import torch
from torch import nn


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 深度空洞卷积
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 逐点卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        # 注意力操作
        return u * attn


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = LKA(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(input.size(), output.size())