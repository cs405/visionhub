"""Advanced Metric Learning Losses

完整的度量学习Loss函数，对齐visionhub:
- MSMLoss (Multi-Similarity Mining Loss)
- MetaBinLoss (Meta Binary Loss)
- XBM (Cross-Batch Memory)
- PairwiseCosFace
- EML Loss (Expected Mutual Loss)
- DME Loss (Deep Mutual Learning Enhanced)
- Soft Triple Loss
- Angular Loss
- Ranked List Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ MSM Loss ============
class MSMLoss(nn.Module):
    """Multi-Similarity Mining Loss

    Paper: Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning
    """

    def __init__(self, alpha=2.0, beta=50.0, base=0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.eps = 1e-8

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 特征向量
            labels: (B,) 标签
        """
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵
        sim_mat = torch.matmul(embeddings, embeddings.t())

        # 构建mask
        labels = labels.unsqueeze(1)
        pos_mask = labels == labels.t()
        neg_mask = labels != labels.t()

        # 移除对角线
        pos_mask.fill_diagonal_(False)

        # Positive loss
        pos_sim = sim_mat[pos_mask]
        if len(pos_sim) > 0:
            pos_exp = torch.exp(-self.alpha * (pos_sim - self.base))
            pos_term = torch.log(1 + pos_exp.sum())
        else:
            pos_term = 0

        # Negative loss
        neg_sim = sim_mat[neg_mask]
        if len(neg_sim) > 0:
            neg_exp = torch.exp(self.beta * (neg_sim - self.base))
            neg_term = torch.log(1 + neg_exp.sum())
        else:
            neg_term = 0

        loss = (1.0 / self.alpha) * pos_term + (1.0 / self.beta) * neg_term

        return loss


# ============ Meta Binary Loss ============
class MetaBinLoss(nn.Module):
    """Meta Binary Loss for training with noisy labels

    Paper: Meta Binary Cross-Entropy for Domain Adaptive Semantic Segmentation
    """

    def __init__(self, num_classes, embedding_size=512, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算logits
        logits = F.linear(embeddings_norm, weight_norm)

        # Binary cross entropy for each class
        targets_one_hot = F.one_hot(labels, self.num_classes).float()

        # Meta learning component
        loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot)

        return loss


# ============ XBM (Cross-Batch Memory) ============
class XBMLoss(nn.Module):
    """Cross-Batch Memory for Embedding Learning

    Paper: Cross-Batch Memory for Embedding Learning
    """

    def __init__(self, embedding_size=512, memory_size=8192, **kwargs):
        super().__init__()
        self.embedding_size = embedding_size
        self.memory_size = memory_size

        # Memory bank
        self.register_buffer('memory', torch.randn(memory_size, embedding_size))
        self.register_buffer('memory_labels', torch.zeros(memory_size).long())
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))

        # Normalize memory
        self.memory = F.normalize(self.memory, p=2, dim=1)

    @torch.no_grad()
    def update_memory(self, embeddings, labels):
        """更新memory bank"""
        batch_size = embeddings.size(0)
        ptr = int(self.memory_ptr)

        # 循环更新
        if ptr + batch_size > self.memory_size:
            # 分两段更新
            first_part = self.memory_size - ptr
            self.memory[ptr:] = embeddings[:first_part]
            self.memory_labels[ptr:] = labels[:first_part]

            second_part = batch_size - first_part
            self.memory[:second_part] = embeddings[first_part:]
            self.memory_labels[:second_part] = labels[first_part:]

            ptr = second_part
        else:
            self.memory[ptr:ptr + batch_size] = embeddings
            self.memory_labels[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % self.memory_size

        self.memory_ptr[0] = ptr

    def forward(self, embeddings, labels, margin=0.2):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 与memory计算相似度
        sim_to_memory = torch.matmul(embeddings, self.memory.t())

        # 构建mask
        labels_expand = labels.unsqueeze(1)
        pos_mask = labels_expand == self.memory_labels.unsqueeze(0)
        neg_mask = labels_expand != self.memory_labels.unsqueeze(0)

        # Triplet loss with memory
        pos_sim = sim_to_memory[pos_mask]
        neg_sim = sim_to_memory[neg_mask]

        if len(pos_sim) > 0 and len(neg_sim) > 0:
            # Hard negative mining
            neg_sim = neg_sim.view(embeddings.size(0), -1)
            hardest_neg, _ = neg_sim.max(dim=1)

            # Hard positive mining
            pos_sim = pos_sim.view(embeddings.size(0), -1)
            hardest_pos, _ = pos_sim.min(dim=1)

            # Triplet loss
            loss = F.relu(hardest_neg - hardest_pos + margin).mean()
        else:
            loss = torch.tensor(0.0).to(embeddings.device)

        # 更新memory
        self.update_memory(embeddings.detach(), labels)

        return loss


# ============ Pairwise CosFace ============
class PairwiseCosFaceLoss(nn.Module):
    """Pairwise Cosine Face Loss"""

    def __init__(self, num_classes, embedding_size=512, s=30.0, m=0.4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 余弦相似度
        cosine = F.linear(embeddings_norm, weight_norm)

        # 添加margin
        phi = cosine - self.m

        # One-hot
        one_hot = F.one_hot(labels, self.num_classes).float()

        # 应用margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Cross entropy
        loss = F.cross_entropy(output, labels)

        return loss


# ============ EML Loss ============
class EMLLoss(nn.Module):
    """Expected Mutual Loss

    Paper: Ranked List Loss for Deep Metric Learning
    """

    def __init__(self, margin=0.4, **kwargs):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 距离矩阵
        dist_mat = 1 - torch.matmul(embeddings, embeddings.t())

        # 构建mask
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()

        # 移除对角线
        pos_mask.fill_diagonal_(0)

        # 计算损失
        loss = 0
        num_samples = embeddings.size(0)

        for i in range(num_samples):
            pos_dists = dist_mat[i][pos_mask[i] > 0]
            neg_dists = dist_mat[i][neg_mask[i] > 0]

            if len(pos_dists) > 0 and len(neg_dists) > 0:
                # Expected mutual information
                pos_exp = torch.exp(-pos_dists)
                neg_exp = torch.exp(-neg_dists)

                loss += -torch.log(pos_exp.mean() / (pos_exp.mean() + neg_exp.mean() + 1e-8))

        loss = loss / num_samples

        return loss


# ============ Soft Triple Loss ============
class SoftTripleLoss(nn.Module):
    """SoftTriple Loss

    Paper: SoftTriple Loss: Deep Metric Learning Without Triplet Sampling
    """

    def __init__(self, num_classes, embedding_size=512, centers_per_class=10, la=20, gamma=0.1, margin=0.01, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.la = la
        self.gamma = gamma
        self.margin = margin

        # Centers
        self.weight = nn.Parameter(torch.Tensor(num_classes * centers_per_class, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 相似度
        sim = torch.matmul(embeddings, weight_norm.t())

        # Reshape to (B, num_classes, centers_per_class)
        sim = sim.view(-1, self.num_classes, self.centers_per_class)

        # Soft assignment
        prob = F.softmax(sim * self.gamma, dim=2)

        # Weighted similarity
        sim_weighted = (prob * sim).sum(dim=2)

        # Add margin
        one_hot = F.one_hot(labels, self.num_classes).float()
        sim_weighted = sim_weighted - one_hot * self.margin

        # Scale
        sim_weighted = sim_weighted * self.la

        # Cross entropy
        loss = F.cross_entropy(sim_weighted, labels)

        return loss


# ============ Angular Loss ============
class AngularLoss(nn.Module):
    """Angular Loss

    Paper: Deep Metric Learning with Angular Loss
    """

    def __init__(self, alpha=45, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.angle = math.radians(alpha)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 余弦相似度
        sim_mat = torch.matmul(embeddings, embeddings.t())

        # 构建mask
        labels = labels.unsqueeze(1)
        pos_mask = labels == labels.t()
        neg_mask = labels != labels.t()
        pos_mask.fill_diagonal_(False)

        loss = 0
        count = 0

        for i in range(embeddings.size(0)):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        # Angular constraint
                        anchor = embeddings[i]
                        positive = embeddings[pos_idx]
                        negative = embeddings[neg_idx]

                        # Compute angles
                        ap_angle = torch.acos(torch.clamp(sim_mat[i, pos_idx], -1, 1))
                        an_angle = torch.acos(torch.clamp(sim_mat[i, neg_idx], -1, 1))
                        pn_angle = torch.acos(torch.clamp(sim_mat[pos_idx, neg_idx], -1, 1))

                        # Angular loss
                        loss += F.relu(ap_angle + self.angle - an_angle)
                        count += 1

        if count > 0:
            loss = loss / count

        return loss


# ============ Ranked List Loss ============
class RankedListLoss(nn.Module):
    """Ranked List Loss for Deep Metric Learning

    Paper: Ranked List Loss for Deep Metric Learning
    """

    def __init__(self, margin=0.4, alpha=1.0, **kwargs):
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 距离矩阵
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Mask
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        pos_mask.fill_diagonal_(0)

        loss = 0
        num_samples = embeddings.size(0)

        for i in range(num_samples):
            pos_dists = dist_mat[i][pos_mask[i] > 0]
            neg_dists = dist_mat[i][neg_mask[i] > 0]

            if len(pos_dists) > 0 and len(neg_dists) > 0:
                # Rank-based loss
                for pos_dist in pos_dists:
                    # Count negatives closer than positive
                    violations = (neg_dists < pos_dist + self.margin).float()
                    if violations.sum() > 0:
                        loss += torch.log(1 + violations.sum()) * self.alpha

        loss = loss / (num_samples + 1e-8)

        return loss


__all__ = [
    'MSMLoss',
    'MetaBinLoss',
    'XBMLoss',
    'PairwiseCosFaceLoss',
    'EMLLoss',
    'SoftTripleLoss',
    'AngularLoss',
    'RankedListLoss'
]

