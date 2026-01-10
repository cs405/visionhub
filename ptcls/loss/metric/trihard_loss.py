"""TriHard Loss

完整迁移自PaddleClas TriHardLoss
Paper: In Defense of the Triplet Loss for Person Re-Identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rerange_index(batch_size, samples_each_class):
    """重排索引以将相同类别的样本聚集在一起"""
    num_classes = batch_size // samples_each_class
    index = []
    for i in range(batch_size):
        for j in range(batch_size):
            class_i = i // samples_each_class
            class_j = j // samples_each_class
            if class_i == class_j:
                index.append(i * batch_size + j)
    return index


class TriHardLoss(nn.Module):
    """TriHard Loss for Person Re-Identification

    Paper: In Defense of the Triplet Loss for Person Re-Identification
    Reference: https://github.com/VisualComputingInstitute/triplet-reid

    基于triplet loss的困难样本挖掘。使用 P * K 采样策略。
    batch_size = P * K，其中P是类别数，K是每类样本数。

    Args:
        batch_size: 批次大小，必须是samples_each_class的倍数
        samples_each_class: 每个类别的样本数，默认为2
        margin: triplet margin，默认为0.1
    """

    def __init__(self, batch_size=120, samples_each_class=2, margin=0.1):
        super().__init__()
        self.margin = margin
        self.samples_each_class = samples_each_class
        self.batch_size = batch_size
        self.rerange_index = torch.LongTensor(rerange_index(batch_size, samples_each_class))

    def _normalize(self, input):
        """L2归一化"""
        input_norm = torch.sqrt(torch.sum(input ** 2, dim=1, keepdim=True))
        return input / (input_norm + 1e-12)

    def forward(self, features, labels=None):
        """
        Args:
            features: (B, D) 特征向量
            labels: (B,) 标签（可选，用于兼容）

        Returns:
            loss: 标量损失值
        """
        batch_size = features.size(0)
        assert batch_size == self.batch_size, f"Batch size {batch_size} != {self.batch_size}"

        # L2归一化
        features = self._normalize(features)
        samples_each_class = self.samples_each_class

        # 计算相似度矩阵（欧氏距离）
        diffs = features.unsqueeze(1) - features.unsqueeze(0)
        similary_matrix = torch.sum(diffs ** 2, dim=-1)

        # 重排索引
        tmp = similary_matrix.view(-1, 1)
        rerange_index = self.rerange_index.to(features.device)
        tmp = tmp[rerange_index]
        similary_matrix = tmp.view(-1, self.batch_size)

        # 分割：忽略自身、正样本、负样本
        ignore = similary_matrix[:, 0:1]
        pos = similary_matrix[:, 1:samples_each_class]
        neg = similary_matrix[:, samples_each_class:]

        # 困难正样本：距离最远的正样本
        hard_pos = torch.max(pos, dim=1)[0]

        # 困难负样本：距离最近的负样本
        hard_neg = torch.min(neg, dim=1)[0]

        # Triplet loss
        loss = F.relu(hard_pos + self.margin - hard_neg)
        loss = torch.mean(loss)

        return loss


__all__ = ['TriHardLoss']

