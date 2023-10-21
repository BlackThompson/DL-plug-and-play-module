# Improving convolutional networks with self-calibrated convolutions (CVPR 2020)
import torch
from torch import nn
import torch.nn.functional as F


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(
            self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.multiply(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    scconv = SCConv(64, 64, stride=1,
                    padding=2, dilation=2, groups=1, pooling_r=4, norm_layer=nn.BatchNorm2d)

    input = torch.rand(1, 64, 64, 64)
    output = scconv(input)
    print(input.size(), output.size())
