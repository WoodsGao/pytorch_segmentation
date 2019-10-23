import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import bn, relu, ResBlock
import math


class AsppPooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            bn(in_features),
            relu,
            nn.Conv2d(in_features, out_features, 1, bias=False),
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.gap(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Aspp(nn.Module):
    def __init__(self, in_features, out_features, rates=[6, 12, 18]):
        super(Aspp, self).__init__()
        conv_list = []
        conv_list.append(
            nn.Sequential(
                bn(in_features),
                relu,
                nn.Conv2d(in_features, out_features, 1, bias=False),
            ))
        conv_list.append(AsppPooling(in_features, out_features))
        for rate in rates:
            conv_list.append(
                nn.Sequential(
                    bn(in_features), relu,
                    nn.Conv2d(in_features,
                              out_features,
                              3,
                              padding=rate,
                              dilation=rate,
                              bias=False)))
        self.conv_list = nn.ModuleList(conv_list)
        self.project = nn.Sequential(
            bn(5 * out_features),
            relu,
            nn.Conv2d(5 * out_features, out_features, 1, bias=False),
            # nn.Dropout(0.5),
        )

    def forward(self, x):
        outputs = []
        for conv in self.conv_list:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        outputs = self.project(outputs)
        return outputs


class DeepLabV3Plus(nn.Module):
    def __init__(self,
                 num_classes,
                 filters=[64, 128, 256, 512, 1024],
                 res_n=[1, 2, 4, 4, 2]):
        super(DeepLabV3Plus, self).__init__()
        assert (len(filters) == 5 and len(res_n) == 5)
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3, bias=False)
        layers = [
            ResBlock(32, filters[0], 2)
        ] + [ResBlock(filters[0], filters[0], se_block=True)] * res_n[0]
        self.res1 = nn.Sequential(*layers)
        layers = [
            ResBlock(filters[0], filters[1], 2)
        ] + [ResBlock(filters[1], filters[1], se_block=True)] * res_n[1]
        self.res2 = nn.Sequential(*layers)
        layers = [
            ResBlock(filters[1], filters[2], 2)
        ] + [ResBlock(filters[2], filters[2], se_block=True)] * res_n[2]
        self.res3 = nn.Sequential(*layers)
        layers = [ResBlock(filters[2], filters[3], 1)
                  ] + [ResBlock(filters[3], filters[3])] * res_n[3]
        self.res4 = nn.Sequential(*layers)
        layers = [ResBlock(filters[3], filters[4], 1)
                  ] + [ResBlock(filters[4], filters[4])] * res_n[4]
        self.res5 = nn.Sequential(*layers)
        self.aspp = Aspp(filters[4], 256)
        self.classifier = nn.Sequential(
            bn(256 + filters[1]), relu,
            nn.Conv2d(256 + filters[1], num_classes, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        low_level_feat = x
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        high_level_feat = x
        high_level_feat = self.aspp(high_level_feat)
        high_level_feat = F.interpolate(high_level_feat,
                                        scale_factor=2,
                                        mode='bilinear',
                                        align_corners=True)
        final_feat = torch.cat([low_level_feat, high_level_feat], 1)
        final_feat = self.classifier(final_feat)
        final_feat = F.interpolate(final_feat,
                                   scale_factor=4,
                                   mode='bilinear',
                                   align_corners=True)
        return final_feat.sigmoid()


class UNet(nn.Module):
    def __init__(self,
                 num_classes,
                 filters=[64, 128, 256, 512, 1024],
                 res_n=[1, 2, 3, 3, 2]):
        super(UNet, self).__init__()
        assert (len(filters) == 5 and len(res_n) == 5)
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3, bias=False)
        layers = [
            ResBlock(32, filters[0], 2)
        ] + [ResBlock(filters[0], filters[0], se_block=True)] * res_n[0]
        self.res1 = nn.Sequential(*layers)
        layers = [
            ResBlock(filters[0], filters[1], 2)
        ] + [ResBlock(filters[1], filters[1], se_block=True)] * res_n[1]
        self.res2 = nn.Sequential(*layers)
        layers = [
            ResBlock(filters[1], filters[2], 2)
        ] + [ResBlock(filters[2], filters[2], se_block=True)] * res_n[2]
        self.res3 = nn.Sequential(*layers)
        layers = [ResBlock(filters[2], filters[3], 2)
                  ] + [ResBlock(filters[3], filters[3])] * res_n[3]
        self.res4 = nn.Sequential(*layers)
        layers = [ResBlock(filters[3], filters[4], 2)
                  ] + [ResBlock(filters[4], filters[4])] * res_n[4]
        self.res5 = nn.Sequential(*layers)
        self.up_conv1 = ResBlock(filters[4], filters[3])
        self.up_conv2 = ResBlock(2 * filters[3], filters[2])
        self.up_conv3 = ResBlock(2 * filters[2], filters[1])
        self.up_conv4 = ResBlock(2 * filters[1], filters[0])
        self.up_conv5 = ResBlock(2 * filters[0], filters[0])
        self.classifier = nn.Sequential(bn(filters[0]), relu,
                                        nn.Conv2d(filters[0], num_classes, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x = F.interpolate(x5,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.up_conv1(x)
        x = torch.cat([x, x4], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.up_conv2(x)
        x = torch.cat([x, x3], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.up_conv3(x)
        x = torch.cat([x, x2], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.up_conv4(x)
        x = torch.cat([x, x1], 1)
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        x = self.up_conv5(x)
        x = self.classifier(x)
        return x.sigmoid()


if __name__ == "__main__":
    model = UNet(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
