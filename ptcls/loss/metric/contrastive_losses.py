"""Contrastive Learning Losses

对比学习损失函数集合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive Loss

    完整版对比损失，用于度量学习

    Args:
        margin: 边界值
        normalize: 是否归一化特征
    """

    def __init__(self, margin=0.5, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 特征向量
            labels: (B,) 标签
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        batch_size = embeddings.size(0)

        # 计算距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # 构建正负样本mask
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()

        # 移除对角线
        mask_pos.fill_diagonal_(0)

        # 正样本loss
        pos_loss = (dist_matrix * mask_pos).sum()

        # 负样本loss
        neg_loss = (F.relu(self.margin - dist_matrix) * mask_neg).sum()

        # 归一化
        num_pos = mask_pos.sum()
        num_neg = mask_neg.sum()

        if num_pos > 0:
            pos_loss = pos_loss / num_pos
        if num_neg > 0:
            neg_loss = neg_loss / num_neg

        loss = pos_loss + neg_loss
        return loss


class SoftSupConLoss(nn.Module):
    """Soft Supervised Contrastive Loss

    软监督对比学习损失
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, soft_labels=None):
        """
        Args:
            features: (B, D) 特征向量
            labels: (B,) 硬标签
            soft_labels: (B, B) 软标签权重矩阵（可选）
        """
        device = features.device
        batch_size = features.size(0)

        # 归一化
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature

        # 数值稳定性
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # 构建mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # 如果提供软标签，使用软标签作为权重
        if soft_labels is not None:
            mask = mask * soft_labels

        # 计算exp
        exp_logits = torch.exp(logits)

        # 移除对角线
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        # Log prob
        log_prob = logits - torch.log((exp_logits * logits_mask).sum(1, keepdim=True) + 1e-12)

        # 计算平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class TripletAngularMarginLoss(nn.Module):
    """Triplet Angular Margin Loss

    带角度边界的三元组损失
    """

    def __init__(self, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 特征向量
            labels: (B,) 标签
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        batch_size = embeddings.size(0)

        # 计算余弦相似度
        similarity = torch.matmul(embeddings, embeddings.T)

        # 构建mask
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()
        mask_pos.fill_diagonal_(0)

        # 困难正样本：相似度最小的正样本
        similarity_pos = similarity * mask_pos + mask_pos.logical_not().float()
        hard_pos = similarity_pos.min(dim=1)[0]

        # 困难负样本：相似度最大的负样本
        similarity_neg = similarity * mask_neg - mask_neg.logical_not().float()
        hard_neg = similarity_neg.max(dim=1)[0]

        # Angular margin
        loss = F.relu(
            torch.acos(torch.clamp(hard_pos, -1 + 1e-7, 1 - 1e-7)) + self.margin -
            torch.acos(torch.clamp(hard_neg, -1 + 1e-7, 1 - 1e-7))
        )

        return loss.mean()


class PairwiseCosFaceLoss(nn.Module):
    """Pairwise CosFace Loss

    成对余弦人脸损失
    """

    def __init__(self, num_classes, embedding_size, s=64.0, m=0.35):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 特征向量
            labels: (B,) 标签
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(embeddings, weight)

        # 应用margin
        one_hot = F.one_hot(labels, self.num_classes).float()
        output = cosine - one_hot * self.m
        output = output * self.s

        # Cross entropy
        loss = F.cross_entropy(output, labels)

        return loss


__all__ = [
    'ContrastiveLoss',
    'SoftSupConLoss',
    'TripletAngularMarginLoss',
    'PairwiseCosFaceLoss',
]

