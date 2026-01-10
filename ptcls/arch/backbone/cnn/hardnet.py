"""Final Niche Backbone Networks

最后的小众骨干网络集合
"""

import torch
import torch.nn as nn


class HarDBlock(nn.Module):
    """HarDNet Block - Harmonic DenseNet"""
    def __init__(self, in_c, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self._get_link(i + 1, in_c, growth_rate, grmul)
            self.links.append(link)

            layers_.append(nn.Sequential(
                nn.BatchNorm2d(inch),
                nn.ReLU(inplace=True),
                nn.Conv2d(inch, outch, 3, 1, 1, bias=False)
            ))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

    def _get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self._get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])

            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]

            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])

        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    """HarDNet - Harmonic DenseNet

    轻量级高效网络
    """

    def __init__(self, num_classes=1000, depth_wise=False):
        super().__init__()

        first_ch = [32, 64]
        second_kernel = 3
        grmul = 1.7
        drop_rate = 0.1

        # Stem
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )

        # Blocks
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]

        blks = []
        for i in range(5):
            blks.append(HarDBlock(first_ch[1] if i == 0 else ch_list[i - 1],
                                 gr[i], grmul, n_layers[i]))

            if i == 0:
                first_ch.append(blks[-1].out_channels)

            if i != 4:
                blks.append(nn.Conv2d(blks[-1].out_channels, ch_list[i], 1))
                if i != 0:
                    blks.append(nn.AvgPool2d(2, 2))

        self.features = nn.Sequential(*blks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(blks[-1].out_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def hardnet68(num_classes=1000):
    """HarDNet-68"""
    return HarDNet(num_classes=num_classes)


__all__ = ['HarDNet', 'hardnet68', 'HarDBlock']

