"""Common CNN Backbones Collection

Includes: DenseNet, VGG, ResNeXt, ShuffleNet, SE-ResNet, etc.
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


# ============ DenseNet ============
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def forward_features(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


@register_backbone('densenet121')
class DenseNet121(DenseNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(32, (6, 12, 24, 16), 64, num_classes=num_classes)
        if pretrained:
            try:
                from torchvision.models import densenet121
                self.load_state_dict(densenet121(pretrained=True).state_dict(), strict=False)
            except: pass


@register_backbone('densenet169')
class DenseNet169(DenseNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(32, (6, 12, 32, 32), 64, num_classes=num_classes)


@register_backbone('densenet201')
class DenseNet201(DenseNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(32, (6, 12, 48, 32), 64, num_classes=num_classes)


# ============ VGG ============
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


@register_backbone('vgg16')
class VGG16(VGG):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(make_vgg_layers(cfgs['vgg16'], batch_norm=True), num_classes)
        if pretrained:
            try:
                from torchvision.models import vgg16_bn
                self.load_state_dict(vgg16_bn(pretrained=True).state_dict(), strict=False)
            except: pass


@register_backbone('vgg19')
class VGG19(VGG):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(make_vgg_layers(cfgs['vgg19'], batch_norm=True), num_classes)


# ============ ShuffleNetV2 ============
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        if self.stride == 1:
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


@register_backbone('shufflenet_v2_x0_5')
@register_backbone('shufflenet_v2')
class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, pretrained=False):
        super().__init__()
        stages_repeats = [4, 8, 4]
        stages_out_channels = [24, 116, 232, 464, 1024]
        stages_out_channels = [int(c * width_mult) for c in stages_out_channels]

        input_channel = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        features = []
        for idxstage, numrepeat in enumerate(stages_repeats):
            output_channel = stages_out_channels[idxstage + 1]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                features.append(ShuffleV2Block(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

        output_channel = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============ ResNeXt ============
class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=32, base_width=4, downsample=None):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=32, width_per_group=4):
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, self.groups, self.base_width, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


@register_backbone('resnext50')
class ResNeXt50(ResNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(ResNeXtBottleneck, [3, 4, 6, 3], num_classes)
        if pretrained:
            try:
                from torchvision.models import resnext50_32x4d
                self.load_state_dict(resnext50_32x4d(pretrained=True).state_dict(), strict=False)
            except: pass


@register_backbone('resnext101')
class ResNeXt101(ResNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(ResNeXtBottleneck, [3, 4, 23, 3], num_classes)

