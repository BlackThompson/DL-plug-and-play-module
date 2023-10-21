import math

import torch.fft
import torch
import torch.nn as nn


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=8, w=5):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x


# 输入 B, N, C,  输出 B, N, C
if __name__ == '__main__':
    block = SpectralGatingNetwork(64).cuda()
    input = torch.rand(1, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
