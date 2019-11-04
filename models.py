import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, lrelu, ResBlock, BLD, Aspp, AsppPooling, DenseBlock
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        # full pre-activation
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
        self.block1 = nn.Sequential(ResBlock(32, 64, stride=2))
        self.block2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=6),
            ResBlock(128, 128),
        )
        self.block3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=6),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=12),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=18),
            ResBlock(256, 512),
            ResBlock(512, 512, dilation=6),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=12),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=18),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=30),
            ResBlock(512, 512),
        )

        self.aspp = Aspp(512, 256)
        self.low_conv = BLD(128, 256)
        self.cls_conv = nn.Sequential(
            bn(512),
            lrelu,
            nn.Conv2d(512, num_classes, 3, 1, 1),
        )
        # init weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, var=None):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        low_level_feat = self.low_conv(x)
        x = self.block3(x)
        x = self.aspp(x)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat([x, low_level_feat], 1)
        x = self.cls_conv(x)
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print(DeepLabV3Plus(8)(a).shape)
