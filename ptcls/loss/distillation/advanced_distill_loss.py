"""Advanced Distillation Losses

完整的蒸馏Loss函数:
- AFD (Attention Feature Distillation)
- PEFD (Patch-wise Enhanced Feature Distillation)
- MGD (Masked Generative Distillation)
- WSL (Wasserstein Self-supervised Learning Loss)
- CCSSL (Cross-modal Contrastive Self-Supervised Learning)
- ReviewKD (Review Knowledge Distillation)
- OFD (Online Feature Distillation)
- CRD (Contrastive Representation Distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ AFD (Attention Feature Distillation) ============
class AFDLoss(nn.Module):
    """Attention Feature Distillation

    Paper: Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    """

    def __init__(self, attention_type='spatial', **kwargs):
        super().__init__()
        self.attention_type = attention_type

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, C, H, W) or list of features
            teacher_features: (B, C, H, W) or list of features
        """
        if isinstance(student_features, (list, tuple)):
            loss = 0
            for s_feat, t_feat in zip(student_features, teacher_features):
                loss += self._compute_loss(s_feat, t_feat)
            return loss / len(student_features)
        else:
            return self._compute_loss(student_features, teacher_features)

    def _compute_loss(self, student_feat, teacher_feat):
        """计算单层loss"""
        if self.attention_type == 'spatial':
            # Spatial attention
            s_attention = self._spatial_attention(student_feat)
            t_attention = self._spatial_attention(teacher_feat)
        elif self.attention_type == 'channel':
            # Channel attention
            s_attention = self._channel_attention(student_feat)
            t_attention = self._channel_attention(teacher_feat)
        else:
            # Both
            s_spatial = self._spatial_attention(student_feat)
            t_spatial = self._spatial_attention(teacher_feat)
            s_channel = self._channel_attention(student_feat)
            t_channel = self._channel_attention(teacher_feat)

            loss_spatial = F.mse_loss(s_spatial, t_spatial)
            loss_channel = F.mse_loss(s_channel, t_channel)
            return loss_spatial + loss_channel

        loss = F.mse_loss(s_attention, t_attention)
        return loss

    def _spatial_attention(self, features):
        """空间注意力"""
        # (B, C, H, W) -> (B, 1, H, W)
        return torch.mean(features ** 2, dim=1, keepdim=True)

    def _channel_attention(self, features):
        """通道注意力"""
        # (B, C, H, W) -> (B, C, 1, 1)
        return torch.mean(features ** 2, dim=[2, 3], keepdim=True)


# ============ PEFD (Patch-wise Enhanced Feature Distillation) ============
class PEFDLoss(nn.Module):
    """Patch-wise Enhanced Feature Distillation"""

    def __init__(self, patch_size=4, **kwargs):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, C, H, W)
            teacher_features: (B, C, H, W)
        """
        B, C, H, W = student_features.shape

        # 分割成patches
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Reshape to patches
        s_patches = student_features.view(B, C, patch_h, self.patch_size, patch_w, self.patch_size)
        s_patches = s_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        s_patches = s_patches.view(-1, C, self.patch_size, self.patch_size)

        t_patches = teacher_features.view(B, C, patch_h, self.patch_size, patch_w, self.patch_size)
        t_patches = t_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        t_patches = t_patches.view(-1, C, self.patch_size, self.patch_size)

        # Compute patch-wise loss
        loss = F.mse_loss(s_patches, t_patches)

        return loss


# ============ MGD (Masked Generative Distillation) ============
class MGDLoss(nn.Module):
    """Masked Generative Distillation

    Paper: Masked Generative Distillation
    """

    def __init__(self, mask_ratio=0.5, **kwargs):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, C, H, W)
            teacher_features: (B, C, H, W)
        """
        B, C, H, W = student_features.shape

        # Generate random mask
        mask = torch.rand(B, 1, H, W).to(student_features.device)
        mask = (mask < self.mask_ratio).float()

        # Apply mask
        masked_student = student_features * mask

        # Reconstruction loss
        loss = F.mse_loss(masked_student, teacher_features * mask)

        # Regularization on unmasked regions
        unmask_loss = F.mse_loss(student_features * (1 - mask), teacher_features * (1 - mask))

        total_loss = loss + 0.1 * unmask_loss

        return total_loss


# ============ WSL (Wasserstein Self-supervised Learning) ============
class WSLLoss(nn.Module):
    """Wasserstein Self-supervised Learning Loss"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        # 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Wasserstein distance (approximated by cosine distance)
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)
        loss = (1 - cosine_sim).mean()

        return loss


# ============ CCSSL (Cross-modal Contrastive Self-Supervised Learning) ============
class CCSSLLoss(nn.Module):
    """Cross-modal Contrastive Self-Supervised Learning"""

    def __init__(self, temperature=0.07, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        B = student_features.size(0)

        # 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Similarity matrix
        logits = torch.matmul(student_norm, teacher_norm.t()) / self.temperature

        # Labels (diagonal elements are positives)
        labels = torch.arange(B).to(student_features.device)

        # Contrastive loss (both directions)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)

        loss = (loss_s2t + loss_t2s) / 2

        return loss


# ============ ReviewKD ============
class ReviewKDLoss(nn.Module):
    """Review Knowledge Distillation

    Paper: Distilling Knowledge via Knowledge Review
    """

    def __init__(self, num_stages=4, **kwargs):
        super().__init__()
        self.num_stages = num_stages

    def forward(self, student_features_list, teacher_features_list):
        """
        Args:
            student_features_list: list of (B, C, H, W)
            teacher_features_list: list of (B, C, H, W)
        """
        loss = 0

        # Hierarchical feature distillation
        for i in range(len(student_features_list)):
            s_feat = student_features_list[i]
            t_feat = teacher_features_list[i]

            # Align channel dimensions if needed
            if s_feat.size(1) != t_feat.size(1):
                # Use adaptive pooling or projection
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.size()[2:])

            # MSE loss
            loss += F.mse_loss(s_feat, t_feat)

            # Review mechanism: current stage guides previous stages
            if i > 0:
                # Upsample current to match previous
                t_feat_up = F.interpolate(t_feat, size=student_features_list[i-1].size()[2:],
                                         mode='bilinear', align_corners=False)
                s_feat_prev = student_features_list[i-1]

                # Additional guidance
                loss += 0.5 * F.mse_loss(s_feat_prev, t_feat_up)

        return loss / len(student_features_list)


# ============ OFD (Online Feature Distillation) ============
class OFDLoss(nn.Module):
    """Online Feature Distillation

    Paper: Feature Distillation: DNN-Oriented JPEG Compression Against Adversarial Examples
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.connector = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        nn.init.xavier_uniform_(self.connector.weight)

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, C_s, H, W)
            teacher_features: (B, C_t, H, W)
        """
        # Align student features to teacher
        student_aligned = self.connector(student_features)

        # Normalize
        student_norm = F.normalize(student_aligned, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Distillation loss
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss


# ============ CRD (Contrastive Representation Distillation) ============
class CRDLoss(nn.Module):
    """Contrastive Representation Distillation

    Paper: Contrastive Representation Distillation
    """

    def __init__(self, embedding_size=128, num_negative=16384, temperature=0.07, **kwargs):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_negative = num_negative
        self.temperature = temperature

        # Negative sample memory bank
        self.register_buffer('memory_bank', torch.randn(num_negative, embedding_size))
        self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        B = student_features.size(0)

        # 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Positive pairs (student-teacher)
        pos_sim = (student_norm * teacher_norm).sum(dim=1, keepdim=True) / self.temperature

        # Negative pairs (student-memory)
        neg_sim = torch.matmul(student_norm, self.memory_bank.t()) / self.temperature

        # Contrastive loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long).to(student_features.device)

        loss = F.cross_entropy(logits, labels)

        return loss


# ============ DML (Deep Mutual Learning) Enhanced ============
class DMELoss(nn.Module):
    """Deep Mutual Learning Enhanced

    Paper: Deep Mutual Learning
    """

    def __init__(self, temperature=4.0, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: (B, num_classes) from model 1
            logits2: (B, num_classes) from model 2
        """
        # KL divergence in both directions
        p1 = F.log_softmax(logits1 / self.temperature, dim=1)
        p2 = F.softmax(logits2 / self.temperature, dim=1)

        loss1 = F.kl_div(p1, p2, reduction='batchmean') * (self.temperature ** 2)

        p2 = F.log_softmax(logits2 / self.temperature, dim=1)
        p1 = F.softmax(logits1 / self.temperature, dim=1)

        loss2 = F.kl_div(p2, p1, reduction='batchmean') * (self.temperature ** 2)

        loss = (loss1 + loss2) / 2

        return loss


__all__ = [
    'AFDLoss',
    'PEFDLoss',
    'MGDLoss',
    'WSLLoss',
    'CCSSLLoss',
    'ReviewKDLoss',
    'OFDLoss',
    'CRDLoss',
    'DMELoss'
]

