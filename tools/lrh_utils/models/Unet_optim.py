from lrh_utils.models.Unet import *
from lrh_utils.models.Resnet_block import *
from torchvision.models import resnet18
from lrh_utils.utils import torchsummary
import torch


# class ResNet18UNet(nn.Module):
#     def __init__(self, n_classes, resnet18_pretrained=True):
#         super(ResNet18UNet, self).__init__()
#         self.n_classes = n_classes
#         self.resnet18_pretrained = resnet18_pretrained
#
#         # change resnet conv1 stride to (1, 1)
#         self.inc = resnet18(pretrained=resnet18_pretrained).conv1
#         self.inc.stride = (1, 1)
#         self.inc = nn.Sequential(
#             self.inc,
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         # out_channel=64, downsample = 1
#         self.down1 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             resnet18(pretrained=resnet18_pretrained).layer1
#         )
#         # out_channel=64, downsample = 2
#         self.down2 = resnet18(self.resnet18_pretrained).layer2
#         # out_channel=128, downsample = 4
#         self.down3 = resnet18(self.resnet18_pretrained).layer3
#         # out_channel=256, downsample = 8
#         self.down4 = resnet18(self.resnet18_pretrained).layer4
#         self.up1 = Up(256 * 3, 256)
#         self.up2 = Up(128 * 3, 128)
#         self.up3 = Up(64 * 3, 64)
#         self.up4 = Up(128, 64)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)  # x1 64
#         x2 = self.down1(x1)  # x2 64
#         x3 = self.down2(x2)  # x3 128
#         x4 = self.down3(x3)  # x4 256
#         x5 = self.down4(x4)  # x5 512
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = torch.sigmoid(self.outc(x))
#         score, _ = torch.max(logits.view(logits.shape[0], logits.shape[1], -1), dim=-1)
#         return logits, score


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, block=BasicBlock):
        super(DownResBlock, self).__init__()
        if block == BasicBlock:
            self.conv1 = BasicBlock(in_channels, out_channels, stride=2, downsample=downsample_conv(in_channels, out_channels))
            self.conv2 = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet18UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResNet18UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownResBlock(64, 128)
        self.down2 = DownResBlock(128, 256)
        self.down3 = DownResBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownResBlock(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.outc = OutConv(512, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x5)
        return logits


if __name__ == "__main__":
    from lrh_utils.utils.torchsummary import summary
    import time

    # model = ResNet18UNet(n_classes=1)
    model = ResNet18UNet(n_channels=3, n_classes=1)
    # # test
    model = model.cuda()
    ts = time.time()
    input_size = (3, 256, 256)
    summary(model=model, input_size=input_size)
    te = time.time()
    print("{:.4f}s".format(te - ts))
