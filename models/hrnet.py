import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.nn import BasicBlock, Bottleneck, ConvNormAct
from pytorch_modules.utils import initialize_weights


class HRModule(nn.Module):
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 multi_scale_output=True):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels,
                             num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)
        initialize_weights(self)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        layers = []
        layers.append(
            block(self.num_inchannels[branch_index],
                  num_channels[branch_index], stride))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index],
                      num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            ConvNormAct(num_inchannels[j], num_inchannels[i],
                                        1),
                            nn.Upsample(scale_factor=2**(j - i),
                                        mode='bilinear',
                                        align_corners=False)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels = num_inchannels[i]
                            convs.append(
                                ConvNormAct(num_inchannels[j],
                                            num_outchannels,
                                            3,
                                            2,
                                            activate=None))

                        else:
                            num_outchannels = num_inchannels[j]
                            convs.append(
                                nn.Sequential(
                                    ConvNormAct(num_inchannels[j],
                                                num_outchannels, 3, 2)))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    def __init__(self, num_classes=2, num_branches_list=[2, 3, 4]):
        self.inplanes = 64
        super(HRNet, self).__init__()
        block = BasicBlock
        # stem net
        self.stem = nn.Sequential(ConvNormAct(3, 64, 3, 2, activate=None),
                                  ConvNormAct(64, 64, 3, 2),
                                  self._make_layer(Bottleneck, 64, 4))

        num_branches = num_branches_list[0]
        num_blocks = [4] * num_branches
        num_inchannels = [
            32 * (2**i) * block.expansion for i in range(num_branches)
        ]
        num_channels = [32 * (2**i) for i in range(num_branches)]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_branches, num_blocks, num_channels, block, num_inchannels)

        num_branches = num_branches_list[1]
        num_blocks = [4] * num_branches
        num_inchannels = [
            32 * (2**i) * block.expansion for i in range(num_branches)
        ]
        num_channels = [32 * (2**i) for i in range(num_branches)]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_branches, num_blocks, num_channels, block, num_inchannels)

        num_branches = num_branches_list[2]
        num_blocks = [4] * num_branches
        num_inchannels = [
            32 * (2**i) * block.expansion for i in range(num_branches)
        ]
        num_channels = [32 * (2**i) for i in range(num_branches)]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            num_branches,
            num_blocks,
            num_channels,
            block,
            num_inchannels,
            multi_scale_output=False)

        self.final_layer = nn.Conv2d(pre_stage_channels[0], num_classes, 1)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        ConvNormAct(num_channels_pre_layer[i],
                                    num_channels_cur_layer[i], 3))

                else:
                    transition_layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    convs.append(ConvNormAct(inchannels, outchannels, 3, 2))
                transition_layers.append(nn.Sequential(*convs))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    num_branches,
                    num_blocks,
                    num_channels,
                    block,
                    num_inchannels,
                    multi_scale_output=True,
                    num_modules=1):

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(num_branches, block, num_blocks, num_inchannels,
                         num_channels, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.stem(x)

        x_list = []
        for i, trans in enumerate(self.transition1):
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i, trans in enumerate(self.transition2):
            if trans is not None:
                x_list.append(trans(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i, trans in enumerate(self.transition3):
            if trans is not None:
                x_list.append(trans(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])
        x = F.interpolate(x,
                          scale_factor=(4, 4),
                          mode='bilinear',
                          align_corners=False)

        return x
