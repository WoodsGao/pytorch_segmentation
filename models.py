import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, relu, ResBlock, BLD, Aspp, AsppPooling
import math


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        # full pre-activation
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
        self.block1 = nn.Sequential(ResBlock(32, 64, stride=2))
        self.block2 = nn.Sequential(
            ResBlock(64, 64, stride=2), 
            ResBlock(64, 64),
            ResBlock(64, 64, dilation=6),
            ResBlock(64, 64),
        )
        self.block3 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=6),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=12),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=18),
            ResBlock(128, 128),
        )
        self.block4 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=6),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=12),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=18),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=30),
            ResBlock(256, 256),
        )
        self.up_conv1 = nn.Sequential(
            ResBlock(256, 128), 
            ResBlock(128, 128), 
            ResBlock(128, 128), 
        )
        self.up_conv2 = nn.Sequential(
            ResBlock(128, 64), 
            ResBlock(64, 64), 
            ResBlock(64, 64), 
        )
        self.up_conv3 = nn.Sequential(
            ResBlock(64, 64), 
            ResBlock(64, 64), 
            ResBlock(64, 64), 
        )
        self.up_conv4 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Dropout(0.5),
            bn(64),
            relu,
            nn.Conv2d(64, num_classes, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        feat1 = x
        x = self.block2(x)
        feat2 = x
        x = self.block3(x)
        feat3 = x
        x = self.block4(x)

        x = x + F.adaptive_avg_pool2d(x, (1, 1))
        x = self.up_conv1(x)

        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = x + feat3
        x = x + F.adaptive_avg_pool2d(feat3, (1, 1))
        x = self.up_conv2(x)

        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = x + feat2
        x = x + F.adaptive_avg_pool2d(feat2, (1, 1))
        x = self.up_conv3(x)

        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = x + feat1
        x = x + F.adaptive_avg_pool2d(feat1, (1, 1))
        x = self.up_conv4(x)

        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return x


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print(DeepLabV3Plus(8)(a).shape)
