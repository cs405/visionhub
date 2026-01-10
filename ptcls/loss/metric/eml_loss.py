"""Ensemble Metric Learning Loss

完整迁移自PaddleClas EmlLoss
Paper: Large Scale Strongly Supervised Ensemble Metric Learning
"""

import torch
import torch.nn as nn
import math


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


class EMLLoss(nn.Module):
    """Ensemble Metric Learning Loss

    Paper: Large Scale Strongly Supervised Ensemble Metric Learning,
           with Applications to Face Verification and Retrieval
    Reference: https://arxiv.org/pdf/1212.6094.pdf

    Args:
        batch_size: 批次大小，必须是samples_each_class的倍数
        samples_each_class: 每个类别的样本数
        thresh: 阈值参数
        beta: beta参数
    """

    def __init__(self, batch_size=40, samples_each_class=2, thresh=20.0, beta=100000):
        super().__init__()
        assert batch_size % samples_each_class == 0
        self.samples_each_class = samples_each_class
        self.batch_size = batch_size
        self.rerange_index = torch.LongTensor(rerange_index(batch_size, samples_each_class))
        self.thresh = thresh
        self.beta = beta

    def surrogate_function(self, beta, theta, bias):
        """代理函数"""
        x = theta * torch.exp(bias)
        output = torch.log(1 + beta * x) / math.log(1 + beta)
        return output

    def surrogate_function_approximate(self, beta, theta, bias):
        """代理函数近似"""
        output = (torch.log(theta) + bias + math.log(beta)) / math.log(1 + beta)
        return output

    def surrogate_function_stable(self, beta, theta, target, thresh):
        """稳定版代理函数"""
        max_gap = torch.tensor(thresh, dtype=torch.float32, device=target.device)

        target_max = torch.maximum(target, max_gap)
        target_min = torch.minimum(target, max_gap)

        loss1 = self.surrogate_function(beta, theta, target_min)
        loss2 = self.surrogate_function_approximate(beta, theta, target_max)
        bias = self.surrogate_function(beta, theta, max_gap)
        loss = loss1 + loss2 - bias
        return loss

    def forward(self, features, labels=None):
        """
        Args:
            features: (B, D) 特征向量
            labels: (B,) 标签（可选，用于兼容）

        Returns:
            loss: 标量损失值
        """
        batch_size = features.size(0)
        samples_each_class = self.samples_each_class

        # 确保batch size正确
        assert batch_size == self.batch_size, f"Batch size {batch_size} != {self.batch_size}"

        # 计算距离矩阵
        diffs = features.unsqueeze(1) - features.unsqueeze(0)
        similary_matrix = torch.sum(diffs ** 2, dim=-1)

        # 重排索引
        tmp = similary_matrix.view(-1, 1)
        rerange_index = self.rerange_index.to(features.device)
        tmp = tmp[rerange_index]
        similary_matrix = tmp.view(-1, batch_size)

        # 分割正样本和负样本
        pos_start = 1
        pos_end = samples_each_class
        neg_start = samples_each_class

        ignore = similary_matrix[:, 0:1]
        pos = similary_matrix[:, pos_start:pos_end]
        neg = similary_matrix[:, neg_start:]

        # 计算正样本损失
        pos_max = torch.max(pos, dim=1, keepdim=True)[0]
        pos = torch.exp(pos - pos_max)
        pos_mean = torch.mean(pos, dim=1, keepdim=True)

        # 计算负样本损失
        neg_min = torch.min(neg, dim=1, keepdim=True)[0]
        neg = torch.exp(neg_min - neg)
        neg_mean = torch.mean(neg, dim=1, keepdim=True)

        # 计算最终损失
        bias = pos_max - neg_min
        theta = neg_mean * pos_mean

        loss = self.surrogate_function_stable(self.beta, theta, bias, self.thresh)
        loss = torch.mean(loss)

        return loss


__all__ = ['EMLLoss']

