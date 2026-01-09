import torch.nn as nn
import torch.nn.functional as F

class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == "relu":
            x = F.relu(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super().__init__()
        self.conv1 = ConvBNLayer(num_channels, num_filters, 3, stride, act="relu")
        self.conv2 = ConvBNLayer(num_filters, num_filters, 3, 1, act=None)
        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(num_channels, num_filters, 1, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if not self.shortcut:
            identity = self.short(identity)
        x = x + identity
        x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super().__init__()
        self.conv1 = ConvBNLayer(num_channels, num_filters, 1, 1, act="relu")
        self.conv2 = ConvBNLayer(num_filters, num_filters, 3, stride, act="relu")
        self.conv3 = ConvBNLayer(num_filters, num_filters * 4, 1, 1, act=None)
        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(num_channels, num_filters * 4, 1, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.shortcut:
            identity = self.short(identity)
        x = x + identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, layers, block_type='bottleneck', class_num=1000):
        super().__init__()
        self.layers = layers
        self.block_type = block_type
        
        self.conv = ConvBNLayer(3, 64, 7, 2, act="relu")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        block = BottleneckBlock if block_type == 'bottleneck' else BasicBlock
        ch_in = 64
        ch_out = 64
        
        self.stages = nn.ModuleList()
        for i, num_layers in enumerate(layers):
            stage = nn.Sequential()
            stride = 1 if i == 0 else 2
            
            # 首个 block 可能需要 shortcut
            stage.add_module("block0", block(ch_in, ch_out, stride, shortcut=False))
            ch_in = ch_out * (4 if block_type == 'bottleneck' else 1)
            
            for j in range(1, num_layers):
                stage.add_module(f"block{j}", block(ch_in, ch_out, 1, shortcut=True))
            
            self.stages.append(stage)
            ch_out *= 2
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.feature_dim = ch_in
        self.fc = nn.Linear(ch_in, class_num)

    def forward_features(self, x):
        x = self.conv(x)
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return logits

def ResNet50(class_num=1000, **kwargs):
    return ResNet([3, 4, 6, 3], block_type='bottleneck', class_num=class_num)

def ResNet18(class_num=1000, **kwargs):
    return ResNet([2, 2, 2, 2], block_type='basic', class_num=class_num)
