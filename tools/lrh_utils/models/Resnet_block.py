from torch import nn

"""
"""


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample_conv(in_channels, out_channels, stride=2, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return nn.Sequential(
        conv1x1(in_channels, out_channels, stride),
        norm_layer(out_channels) if not norm_layer == nn.GroupNorm else norm_layer(16, out_channels)
    )


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None, act_func=None):
        super(BasicBlock, self).__init__()
        if norm_layer == None:
            norm_layer = nn.BatchNorm2d
        assert norm_layer in [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm]
        if act_func == None:
            act_func = nn.ReLU(inplace=True)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel) if not norm_layer == nn.GroupNorm else norm_layer(16, out_channel)
        self.act_func = act_func
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel) if not norm_layer == nn.GroupNorm else norm_layer(16, out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_func(out)

        return out
