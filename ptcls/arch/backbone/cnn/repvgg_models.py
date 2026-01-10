"""RepVGG Implementation
        super().__init__([8, 14, 24, 1], [2.5, 2.5, 2.5, 5], use_se=True, num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_D2se(RepVGG):
@register_backbone('repvgg_d2se')


        super().__init__([4, 6, 16, 1], [3, 3, 3, 5], g4_map, num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B3g4(RepVGG):
@register_backbone('repvgg_b3g4')


        super().__init__([4, 6, 16, 1], [3, 3, 3, 5], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B3(RepVGG):
@register_backbone('repvgg_b3')


        super().__init__([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], g4_map, num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B2g4(RepVGG):
@register_backbone('repvgg_b2g4')


        super().__init__([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B2(RepVGG):
@register_backbone('repvgg_b2')


        super().__init__([4, 6, 16, 1], [2, 2, 2, 4], g4_map, num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B1g4(RepVGG):
@register_backbone('repvgg_b1g4')


        super().__init__([4, 6, 16, 1], [2, 2, 2, 4], g2_map, num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B1g2(RepVGG):
@register_backbone('repvgg_b1g2')


        super().__init__([4, 6, 16, 1], [2, 2, 2, 4], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B1(RepVGG):
@register_backbone('repvgg_b1')


        super().__init__([4, 6, 16, 1], [1, 1, 1, 2.5], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_B0(RepVGG):
@register_backbone('repvgg_b0')


        super().__init__([2, 4, 14, 1], [1.5, 1.5, 1.5, 2.75], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_A2(RepVGG):
@register_backbone('repvgg_a2')


        super().__init__([2, 4, 14, 1], [1, 1, 1, 2.5], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_A1(RepVGG):
@register_backbone('repvgg_a1')


        super().__init__([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5], num_classes=num_classes)
    def __init__(self, num_classes=1000, pretrained=False):
class RepVGG_A0(RepVGG):
@register_backbone('repvgg_a0')


g4_map = {l: 4 for l in optional_groupwise_layers}
g2_map = {l: 2 for l in optional_groupwise_layers}
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]


        return x
        x = self.linear(x)
        x = x.view(x.size(0), -1)
        x = self.gap(x)
        x = self.stage4(x)
        x = self.stage3(x)
        x = self.stage2(x)
        x = self.stage1(x)
        x = self.stage0(x)
    def forward(self, x):
    
        return nn.Sequential(*blocks)
            self.cur_layer_idx += 1
            self.in_planes = planes
                                     padding=1, groups=groups, use_se=self.use_se))
            blocks.append(RepVGGBlock(self.in_planes, planes, kernel_size=3, stride=stride,
            groups = self.group_map.get(self.cur_layer_idx, 1)
        for stride in strides:
        blocks = []
        strides = [stride] + [1] * (num_blocks - 1)
    def _make_stage(self, planes, num_blocks, stride):
        
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage0 = RepVGGBlock(3, self.in_planes, kernel_size=3, stride=2, padding=1)
        
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        
        self.group_map = group_map or {}
        self.cur_layer_idx = 1
        self.use_se = use_se
        
        assert len(width_multiplier) == 4
        
        super().__init__()
                 use_se=False, num_classes=1000):
    def __init__(self, num_blocks, width_multiplier=None, group_map=None, 
    """RepVGG Network"""
class RepVGG(nn.Module):


        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + id_out))
        
            id_out = self.rbr_identity(x)
        else:
            id_out = 0
        if self.rbr_identity is None:
    def forward(self, x):
        
        )
            nn.BatchNorm2d(out_channels)
            nn.Conv2d(in_channels, out_channels, 1, stride, padding - kernel_size // 2, groups=groups, bias=False),
        self.rbr_1x1 = nn.Sequential(
        )
            nn.BatchNorm2d(out_channels)
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        self.rbr_dense = nn.Sequential(
        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        
            self.se = nn.Identity()
        else:
            self.se = SEBlock(out_channels, out_channels // 16)
        if use_se:
        
        self.nonlinearity = nn.ReLU()
        
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        super().__init__()
                 padding=1, groups=1, use_se=False):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
    """RepVGG Block with reparameterization"""
class RepVGGBlock(nn.Module):


        return x * out
        out = torch.sigmoid(out)
        out = self.up(out)
        out = F.relu(out)
        out = self.down(out)
        out = F.adaptive_avg_pool2d(x, 1)
    def forward(self, x):

        self.up = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        super().__init__()
    def __init__(self, input_channels, internal_neurons):
    """Squeeze-and-Excitation Block"""
class SEBlock(nn.Module):


from ..registry import register_backbone
import torch.nn.functional as F
import torch.nn as nn
import torch

"""
https://arxiv.org/abs/2101.03697
Paper: RepVGG: Making VGG-style ConvNets Great Again


