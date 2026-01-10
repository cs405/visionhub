"""RepVGG: Making VGG-style ConvNets Great Again

完整迁移自PaddleClas RepVGG
Paper: RepVGG: Making VGG-style ConvNets Great Again
Reference: https://arxiv.org/abs/2101.03697
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        self.input_channels = input_channels

    def forward(self, x):
        identity = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return identity * x


class RepVGGBlock(nn.Module):
    """RepVGG Block with structural re-parameterization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        use_se=False,
        deploy=False
    ):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            # 部署模式：只有一个融合的3x3卷积
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias=True
            )
        else:
            # 训练模式：3个分支
            # 1. 3x3卷积
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # 2. 1x1卷积
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # 3. Identity (仅当输入输出通道相同且stride=1时)
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.se(self.rbr_reparam(x)))

        # 训练模式：3个分支相加
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        return self.nonlinearity(self.se(out))

    def get_equivalent_kernel_bias(self):
        """获取等效的卷积核和偏置（用于重参数化）"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # 将1x1卷积核填充到3x3
        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])

        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        """融合BN到卷积"""
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # Identity BN
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """切换到部署模式"""
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # 删除训练模式的分支
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

        self.deploy = True


class RepVGG(nn.Module):
    """RepVGG Network

    Args:
        num_blocks: 每个stage的block数量
        width_multiplier: 宽度缩放因子
        num_classes: 分类数
        use_se: 是否使用SE模块
        deploy: 是否为部署模式
    """

    def __init__(
        self,
        num_blocks,
        width_multiplier=None,
        num_classes=1000,
        use_se=False,
        deploy=False
    ):
        super().__init__()

        if width_multiplier is None:
            width_multiplier = [0.75, 0.75, 0.75, 2.5]

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.use_se = use_se

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            3, self.in_planes, kernel_size=3, stride=2,
            padding=1, use_se=use_se, deploy=deploy
        )
        self.cur_layer_idx = 1

        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        """构建一个stage"""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(
                RepVGGBlock(
                    self.in_planes, planes, kernel_size=3,
                    stride=stride, padding=1,
                    use_se=self.use_se, deploy=self.deploy
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_features(self, x):
        """提取特征"""
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


def repvgg_a0(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-A0"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_a1(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-A1"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_a2(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-A2"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_b0(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-B0"""
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_b1(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-B1"""
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_b2(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-B2"""
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


def repvgg_b3(num_classes=1000, deploy=False, **kwargs):
    """RepVGG-B3"""
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        num_classes=num_classes,
        deploy=deploy,
        **kwargs
    )


__all__ = [
    'RepVGG',
    'repvgg_a0',
    'repvgg_a1',
    'repvgg_a2',
    'repvgg_b0',
    'repvgg_b1',
    'repvgg_b2',
    'repvgg_b3',
]

