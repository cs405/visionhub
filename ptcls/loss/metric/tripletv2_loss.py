"""Triplet Loss V2

Enhanced triplet loss with hard sample mining
Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossV2(nn.Module):
    """Triplet Loss V2 with Hard Positive/Negative Mining

    Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering
    Reference: https://arxiv.org/pdf/1503.03832.pdf

    使用困难正负样本挖掘的三元组损失

    Args:
        margin: 三元组边界
        normalize_feature: 是否归一化特征
    """

    def __init__(self, margin=0.5, normalize_feature=True):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels):
        """
        Args:
            inputs: (B, D) 特征矩阵
            labels: (B,) 标签

        Returns:
            loss: 标量损失值
        """
        # 归一化
        if self.normalize_feature:
            inputs = F.normalize(inputs, p=2, dim=-1)

        bs = inputs.size(0)

        # 计算欧氏距离矩阵
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(bs, bs)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = torch.clamp(dist, min=1e-12).sqrt()

        # 构建正负样本mask
        labels_expand = labels.unsqueeze(0).expand(bs, bs)
        is_pos = labels_expand == labels_expand.t()
        is_neg = labels_expand != labels_expand.t()

        # 困难正样本挖掘：找最远的正样本
        dist_ap = dist[is_pos].view(bs, -1).max(dim=1, keepdim=True)[0]

        # 困难负样本挖掘：找最近的负样本
        dist_an = dist[is_neg].view(bs, -1).min(dim=1, keepdim=True)[0]

        # squeeze
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        # Ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


__all__ = ['TripletLossV2']

