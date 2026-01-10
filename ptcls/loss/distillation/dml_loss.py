"""Deep Mutual Learning Loss

Implementation of deep mutual learning
Paper: Deep Mutual Learning
"""

import torch
import torch.nn as nn


class DMLLoss(nn.Module):
    """Deep Mutual Learning Loss

    Paper: Deep Mutual Learning
    用于模型之间的互学习，通过KL散度实现双向知识传递

    Args:
        act: 激活函数类型 ('softmax', 'sigmoid', None)
        sum_across_class_dim: 是否在类别维度求和
        eps: 数值稳定性参数
    """

    def __init__(self, act="softmax", sum_across_class_dim=False, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.sum_across_class_dim = sum_across_class_dim

        if act is not None:
            assert act in ["softmax", "sigmoid"], f"Unsupported activation: {act}"

        if act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def _kldiv(self, x, target):
        """计算KL散度"""
        class_num = x.size(-1)
        cost = target * torch.log((target + self.eps) / (x + self.eps)) * class_num
        return cost

    def forward(self, x, target, gt_label=None):
        """
        Args:
            x: (B, C) 模型1的输出
            target: (B, C) 模型2的输出（作为目标）
            gt_label: (B, 2, C) 可选，用于多标签DML损失

        Returns:
            loss: 标量损失值
        """
        # 应用激活函数
        if self.act is not None:
            x = self.act(x)
            target = self.act(target)

        # 双向KL散度
        loss = self._kldiv(x, target) + self._kldiv(target, x)
        loss = loss / 2

        # 多标签DML损失
        if gt_label is not None:
            # gt_label: [actual_labels, label_ratios]
            gt_label_actual = gt_label[:, 0, :]
            label_ratio = gt_label[:, 1, :]

            # 创建mask
            targets_mask = (gt_label_actual > 0.5).float()

            # 计算权重
            weight = self._ratio2weight(targets_mask, label_ratio)
            weight = weight * (gt_label_actual > -1).float()

            loss = loss * weight

        # 聚合
        if self.sum_across_class_dim:
            loss = loss.sum(1).mean()
        else:
            loss = loss.mean()

        return loss

    def _ratio2weight(self, targets_mask, label_ratio):
        """将比率转换为权重"""
        # 简化实现：根据正负样本比例调整权重
        pos_num = targets_mask.sum(dim=1, keepdim=True)
        neg_num = (1 - targets_mask).sum(dim=1, keepdim=True)

        weight = torch.ones_like(targets_mask)

        # 正样本权重
        weight = torch.where(
            targets_mask > 0.5,
            label_ratio.unsqueeze(1).expand_as(targets_mask),
            weight
        )

        # 负样本权重
        weight = torch.where(
            targets_mask <= 0.5,
            (1 - label_ratio.unsqueeze(1).expand_as(targets_mask)),
            weight
        )

        return weight


__all__ = ['DMLLoss']

