from torch import nn
import torch
from torch.nn import functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class Conv2dCircle(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, stride, h_circle=True):
        super(Conv2dCircle, self).__init__()
        """

        :param in_channel:
        :param out_channel:
        :param ksize:
        :param stride:
        :param h_circle: bool 横向拼接
        """
        self.circle_padding = lambda x: torch.cat([x[:, :, :, -(ksize // 2):], x, x[:, :, :, :ksize // 2]],
                                                  dim=-1) if h_circle else \
            torch.cat([x[:, :, -(ksize // 2):, :], x, x[:, :, :ksize // 2, :]], dim=-2)
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, stride, (ksize // 2, 0) if h_circle else (0, ksize // 2))

        # self.conv = nn.Conv2d(in_channel, out_channel, ksize, stride, ksize//2)

    def forward(self, x):
        x = self.circle_padding(x)
        x = self.conv(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
