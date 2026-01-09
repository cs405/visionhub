"""Metric Learning Loss Functions

Includes: ArcFace, CosFace, SphereFace, CenterLoss, ContrastiveLoss, NPairsLoss, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ ArcFace ============
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, s=64.0, m=0.5):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        # Compute theta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Add angular margin
        target_theta = theta.scatter(1, labels.unsqueeze(1), 0)
        target_theta = target_theta.gather(1, labels.unsqueeze(1)).squeeze(1)
        target_theta = target_theta + self.m

        # Convert back to cosine
        target_cosine = torch.cos(target_theta)
        cosine = cosine * 1.0
        cosine.scatter_(1, labels.unsqueeze(1), target_cosine.unsqueeze(1))

        # Scale
        output = cosine * self.s

        return F.cross_entropy(output, labels)


# ============ CosFace ============
class CosFaceLoss(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, s=64.0, m=0.35):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        # Add cosine margin
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)

        output = (one_hot * (cosine - self.m)) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, labels)


# ============ SphereFace ============
class SphereFaceLoss(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, s=64.0, m=4):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize weight
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), weight)

        # Compute theta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Compute m*theta
        target_theta = theta.gather(1, labels.unsqueeze(1))
        target_theta = target_theta * self.m

        # Convert back
        target_cosine = torch.cos(target_theta)
        cosine.scatter_(1, labels.unsqueeze(1), target_cosine)

        output = cosine * self.s
        return F.cross_entropy(output, labels)


# ============ CenterLoss ============
class CenterLoss(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=512, lambda_c=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Compute distance
        distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(embeddings, self.centers.t(), beta=1, alpha=-2)

        # Select distances for the correct classes
        classes = torch.arange(self.num_classes, device=labels.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss * self.lambda_c


# ============ ContrastiveLoss ============
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1: (B, D)
            embeddings2: (B, D)
            labels: (B,) 1 for similar, 0 for dissimilar
        """
        distances = F.pairwise_distance(embeddings1, embeddings2)

        losses = labels.float() * torch.pow(distances, 2) + \
                 (1 - labels).float() * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        return losses.mean()


# ============ NPairsLoss ============
class NPairsLoss(nn.Module):
    def __init__(self, l2_reg=0.02):
        super().__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives, labels):
        """
        Args:
            anchors: (B, D)
            positives: (B, D)
            labels: (B,)
        """
        # L2 normalize
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(anchors, positives.t())

        # Create labels for cross entropy
        batch_size = anchors.size(0)
        labels_remapped = torch.arange(batch_size, device=anchors.device)

        loss = F.cross_entropy(similarity_matrix, labels_remapped)

        # Add L2 regularization
        if self.l2_reg > 0:
            loss = loss + self.l2_reg * (torch.pow(anchors, 2).sum() + torch.pow(positives, 2).sum()) / batch_size

        return loss


# ============ TripletAngularMarginLoss ============
class TripletAngularMarginLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_factor=0.5):
        super().__init__()
        self.margin = margin
        self.lambda_factor = lambda_factor

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (B, D)
            positive: (B, D)
            negative: (B, D)
        """
        # Angular distance
        def angular_distance(x1, x2):
            cos_sim = F.cosine_similarity(x1, x2)
            angle = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7))
            return angle

        ap_dist = angular_distance(anchor, positive)
        an_dist = angular_distance(anchor, negative)

        loss = torch.clamp(ap_dist - an_dist + self.margin, min=0.0)

        return loss.mean()


# ============ FocalLoss ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ============ LabelSmoothingCrossEntropy ============
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)

        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes

        loss = (-targets_smooth * log_probs).sum(dim=1)

        return loss.mean()


# ============ SoftTargetCrossEntropy ============
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) soft targets (probabilities)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        loss = torch.sum(-targets * log_probs, dim=1)

        return loss.mean()


# Export all
__all__ = [
    'ArcFaceLoss',
    'CosFaceLoss',
    'SphereFaceLoss',
    'CenterLoss',
    'ContrastiveLoss',
    'NPairsLoss',
    'TripletAngularMarginLoss',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'SoftTargetCrossEntropy',
]

