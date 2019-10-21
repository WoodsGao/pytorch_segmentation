import torch
import torch.nn as nn
import torch.nn.functional as F

relu = nn.LeakyReLU(0.1)
bn = nn.BatchNorm2d


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            bn(filters),
            nn.Conv2d(filters, filters // 16, 1, bias=False),
            relu,
            nn.Conv2d(filters // 16, filters, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            bn(in_features),
            relu,
            # SELayer(in_features),
            nn.Conv2d(in_features, out_features // 2, 1, 1, 0, bias=False),
            bn(out_features // 2),
            relu,
            nn.Conv2d(out_features // 2,
                      out_features,
                      3,
                      stride,
                      dilation,
                      bias=False,
                    #   groups=32 if out_features % 32 == 0 else 1,
                      dilation=dilation),
            # SELayer(out_features),
        )
        self.downsample = None
        if stride > 1 or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, 3, stride, 1), )

    def forward(self, x):
        if self.downsample is not None:
            downsample = self.downsample(x)
        else:
            downsample = x
        return downsample + self.block(x)


class AsppPooling(nn.Module):
    def __init__(self, in_features, out_features, rates=[6, 12, 18]):
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
                 filters=[64, 128, 256, 256, 512],
                 res_n=[1, 1, 1,1,1]):
        super(DeepLabV3Plus, self).__init__()
        assert (len(filters) == 5 and len(res_n) == 5)
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3, bias=False)
        layers = [ResBlock(32, filters[0], 2, dilation=2)
                  ] + [ResBlock(filters[0], filters[0])] * res_n[0]
        self.res1 = nn.Sequential(*layers)
        layers = [ResBlock(filters[0], filters[1], 2, dilation=2)
                  ] + [ResBlock(filters[1], filters[1])] * res_n[1]
        self.res2 = nn.Sequential(*layers)
        layers = [ResBlock(filters[1], filters[2], 2, dilation=2)
                  ] + [ResBlock(filters[2], filters[2])] * res_n[2]
        self.res3 = nn.Sequential(*layers)
        layers = [ResBlock(filters[2], filters[3], dilation=2)
                  ] + [ResBlock(filters[3], filters[3])] * res_n[3]
        self.res4 = nn.Sequential(*layers)
        layers = [ResBlock(filters[3], filters[4], dilation=2)
                  ] + [ResBlock(filters[4], filters[4])] * res_n[4]
        self.res5 = nn.Sequential(*layers)
        self.aspp = Aspp(filters[4], 256)
        self.classifier = ResBlock(256 + filters[1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.size()
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
                                        low_level_feat.size()[2:],
                                        mode='bilinear',
                                        align_corners=True)
        final_feat = torch.cat([low_level_feat, high_level_feat], 1)
        final_feat = self.classifier(final_feat)
        final_feat = F.interpolate(final_feat,
                                   size[2:],
                                   mode='bilinear',
                                   align_corners=True)
        return final_feat.sigmoid()


if __name__ == "__main__":
    model = DeepLabV3Plus(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
