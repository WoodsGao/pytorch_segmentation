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
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=30),
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
        self.aspp_pooling = AsppPooling(512, 512)
        self.up_conv = BLD(1024, 512, 1)
        self.low_conv = BLD(64, 512, 1)
        self.middle_conv = BLD(128, 512, 1)
        self.obj_conv = nn.Sequential(
            ResBlock(1024, 256),
            ResBlock(256,512),
            bn(512),
            relu,
            nn.Conv2d(512, 1, 3, padding=1),
        )
        self.pre_cls_conv = nn.Sequential(
            ResBlock(1024, 256), 
            ResBlock(256, 512)
        )
        self.cls_conv = nn.Sequential(
            # nn.Dropout(0.1),
            bn(512),
            relu,
            nn.Conv2d(512, num_classes - 1, 3, padding=1),
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
        low = self.low_conv(x)
        x = self.block2(x)
        middle = self.middle_conv(x)
        x = self.block3(x)

        x = torch.cat([x, self.aspp_pooling(x)], 1)
        x = self.up_conv(x)

        cls_mask = F.interpolate(x,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        cls_mask = torch.cat([cls_mask, middle], 1)
        cls_mask = self.pre_cls_conv(cls_mask)
        obj_mask = F.interpolate(cls_mask,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        cls_mask = self.cls_conv(cls_mask)
        cls_mask = F.interpolate(cls_mask,
                                 scale_factor=4,
                                 mode='bilinear',
                                 align_corners=True)

        obj_mask = torch.cat([obj_mask, low], 1)
        obj_mask = self.obj_conv(obj_mask)
        obj_mask = F.interpolate(obj_mask,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        return obj_mask.sigmoid(), cls_mask


if __name__ == "__main__":
    a = torch.ones([2, 3, 224, 224])
    print([o.shape for o in DeepLabV3Plus(8)(a)])
