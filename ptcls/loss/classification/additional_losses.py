"""Additional Loss Functions

包括：MultiLabel Loss, Deep Hash Loss, WSL Loss等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============ Multi-Label Loss ============
class MultiLabelLoss(nn.Module):
    """Multi-Label Classification Loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw scores
            targets: (B, C) binary labels
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=self.reduction)
        return loss


# ============ Asymmetric Loss (for Multi-Label) ============
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: (B, C) logits
            y: (B, C) binary targets
        """
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            loss = -(los_pos + los_neg)

        return loss.mean()


# ============ Deep Hash Loss ============
class DeepHashLoss(nn.Module):
    """Loss for deep hashing (binary code learning)"""
    def __init__(self, num_bits=48, lambda_q=0.01):
        super().__init__()
        self.num_bits = num_bits
        self.lambda_q = lambda_q

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, num_bits) continuous embeddings
            labels: (B,) class labels
        """
        # Quantization loss (encourage binary values)
        quantization_loss = torch.mean((torch.abs(embeddings) - 1) ** 2)

        # Similarity loss
        similarity_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Hamming distance
        embeddings_norm = torch.sign(embeddings)
        dist = 0.5 * (self.num_bits - embeddings_norm @ embeddings_norm.t())

        # Weighted distance loss
        similar_loss = similarity_matrix * dist
        dissimilar_loss = (1 - similarity_matrix) * torch.clamp(self.num_bits / 2 - dist, min=0)

        pair_loss = (similar_loss + dissimilar_loss).mean()

        total_loss = pair_loss + self.lambda_q * quantization_loss
        return total_loss


# ============ Pairwise Ranking Loss ============
class PairwiseRankingLoss(nn.Module):
    """Ranking loss for retrieval"""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (B, D)
            positive: (B, D)
            negative: (B, D)
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


# ============ Angular Softmax Loss (SphereFace) ============
class AngularSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalize weights
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(input, p=2, dim=1)

        # Compute cosine
        wf = x @ w.t()

        # Add angular margin
        numerator = torch.diagonal(wf.transpose(0, 1)[label])
        theta = torch.acos(torch.clamp(numerator, -1.0 + 1e-7, 1.0 - 1e-7))

        # Apply m*theta
        numerator = torch.cos(self.m * theta)

        # Update logits
        wf_label = wf.clone()
        wf_label[torch.arange(input.size(0)), label] = numerator

        loss = F.cross_entropy(wf_label, label)
        return loss


# ============ Large Margin Cosine Loss (CosFace variant) ============
class LargeMarginCosineLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalize
        x = F.normalize(input, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine
        cosine = F.linear(x, w)

        # Add margin to target class
        phi = cosine - self.m

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, label)
        return loss


# ============ Proxy-NCA Loss ============
class ProxyNCALoss(nn.Module):
    """Proxy Neighborhood Component Analysis Loss"""
    def __init__(self, num_classes, embedding_size, smoothing_const=0.1, temperature=3.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.smoothing_const = smoothing_const
        self.temperature = temperature

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        # Compute distances to all proxies
        dist = embeddings @ proxies.t()  # (B, num_classes)

        # Temperature scaling
        dist = dist / self.temperature

        # NCA loss
        pos_dist = dist[torch.arange(len(labels)), labels]

        # Exclude positive proxy from denominator
        mask = torch.ones_like(dist)
        mask[torch.arange(len(labels)), labels] = 0

        neg_exp_sum = (torch.exp(dist) * mask).sum(dim=1)
        loss = -torch.log(torch.exp(pos_dist) / (torch.exp(pos_dist) + neg_exp_sum + self.smoothing_const))

        return loss.mean()


# ============ Lifted Structure Loss ============
class LiftedStructureLoss(nn.Module):
    """Lifted Structure Loss for deep metric learning"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # Compute pairwise distances
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = 2 - 2 * embeddings @ embeddings.t()

        # Create masks
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()

        # Positive pairs
        pos_pairs = dist_mat * mask_pos

        # Negative pairs
        neg_exp = torch.exp(self.margin - dist_mat) * mask_neg
        neg_log = torch.log(neg_exp.sum(dim=1, keepdim=True) + 1e-16)

        # Combine
        loss_mat = torch.relu(pos_pairs + neg_log)
        loss = (loss_mat * mask_pos).sum() / (mask_pos.sum() + 1e-16)

        return loss


# ============ Histogram Loss ============
class HistogramLoss(nn.Module):
    """Histogram Loss for deep metric learning"""
    def __init__(self, num_steps=100, margin=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Pairwise similarities
        sim_mat = embeddings @ embeddings.t()

        # Masks
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()

        # Remove diagonal
        mask_pos = mask_pos - torch.eye(len(labels), device=embeddings.device)

        # Histogram bins
        sim_pos = sim_mat[mask_pos.bool()]
        sim_neg = sim_mat[mask_neg.bool()]

        # Compute histograms
        pos_hist = torch.histc(sim_pos, bins=self.num_steps, min=-1, max=1)
        neg_hist = torch.histc(sim_neg, bins=self.num_steps, min=-1, max=1)

        # Normalize
        pos_hist = pos_hist / (pos_hist.sum() + 1e-16)
        neg_hist = neg_hist / (neg_hist.sum() + 1e-16)

        # Compute overlap loss
        overlap = torch.min(pos_hist, neg_hist).sum()
        loss = overlap + torch.relu(self.margin - (sim_pos.mean() - sim_neg.mean()))

        return loss


# Export all
__all__ = [
    'MultiLabelLoss',
    'AsymmetricLoss',
    'DeepHashLoss',
    'PairwiseRankingLoss',
    'AngularSoftmaxLoss',
    'LargeMarginCosineLoss',
    'ProxyNCALoss',
    'LiftedStructureLoss',
    'HistogramLoss',
]

