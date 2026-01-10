"""Soft Target Cross Entropy Loss and GoogLeNet Loss

完整迁移自PaddleClas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """Soft Target Cross Entropy Loss

    用于知识蒸馏等场景，目标是软标签（概率分布）而非硬标签
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        """
        Args:
            x: (B, C) 模型输出logits
            target: (B, C) 软目标（概率分布）

        Returns:
            loss: 标量损失值
        """
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        loss = loss.mean()
        return loss


class GoogLeNetLoss(nn.Module):
    """GoogLeNet Loss

    GoogLeNet使用多个辅助分类器的损失函数
    Paper: Going Deeper with Convolutions
    Reference: https://arxiv.org/pdf/1409.4842v1.pdf

    Args:
        epsilon: label smoothing参数（GoogLeNet不支持）
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon > 0 and epsilon < 1):
            raise ValueError("GoogLeNet is not support label_smooth")

    def forward(self, inputs, labels):
        """
        Args:
            inputs: 包含3个输出的tuple/list [main_output, aux1_output, aux2_output]
            labels: (B,) 标签

        Returns:
            loss: 标量损失值
        """
        input0, input1, input2 = inputs

        # 处理dict格式
        if isinstance(input0, dict):
            input0 = input0["logits"]
        if isinstance(input1, dict):
            input1 = input1["logits"]
        if isinstance(input2, dict):
            input2 = input2["logits"]

        # 计算三个分类器的损失
        loss0 = F.cross_entropy(input0, labels)
        loss1 = F.cross_entropy(input1, labels)
        loss2 = F.cross_entropy(input2, labels)

        # 加权组合：主分类器权重1.0，辅助分类器权重0.3
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2
        loss = loss.mean()

        return loss


__all__ = ['SoftTargetCrossEntropy', 'GoogLeNetLoss']

