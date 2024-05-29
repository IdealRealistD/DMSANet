import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, inplanes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(inplanes, inplanes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(inplanes // ratio, inplanes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, size=3, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=size, stride=stride, padding=size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(1/5.5, inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=size, stride=1, padding=size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out = out + identity
        out = self.relu(out)

        del x
        del identity

        return out


class BasicBlock_2(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, size=3, downsample=None):
        super(BasicBlock_2, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=size, stride=stride, padding=size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(1/5.5, inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=size, stride=1, padding=size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out = out + identity
        out = self.relu(out)

        del x
        del identity

        return out


class ECGNet(nn.Module):
    def __init__(self, input_channel=64, num_classes=15):
        super(ECGNet, self).__init__()
        self.in_channel = 64
        blocks_num = [3, 4]
        sizes = [
            [7, 7],
            [9, 9],
        ]
        self.sizes = sizes
        layers = [
            [3, 2],
            [3, 2],
        ]
        self.conv1 = nn.Conv1d(12, input_channel, kernel_size=17, stride=2, padding=8, bias=False)
        self.bn1 = nn.BatchNorm1d(input_channel)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.LeakyReLU(1/5.5, inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, blocks_num[0], stride=1, size=5)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks_num[1], stride=2, size=5)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512 * len(sizes), num_classes)

        self.layers1_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.layers = nn.Sequential()
            self.in_channel = 128
            self.layers.add_module('layer{}_1'.format(size),
                                   self._make_layer(BasicBlock_2, 256, layers[i][0], stride=2, size=sizes[i][0]))
            self.layers.add_module('layer{}_2'.format(size),
                                   self._make_layer(BasicBlock_2, 512, layers[i][1], stride=2, size=sizes[i][1]))
            self.layers1_list.append(self.layers)

            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def _make_layer(self, block, channel, block_num, stride=1, size=3):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            size=size,
                            stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, size=size))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.max_pool_1(x0)
        x0 = self.layer1(x0)
        x0 = self.layer2(x0)
        x0 = self.max_pool_2(x0)

        xs = []
        for i in range(len(self.sizes)):
            x = self.layers1_list[i](x0)
            x = self.global_avg_pool(x)
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        del x0
        del xs
        del x

        return out


ECGnet = ECGNet()
