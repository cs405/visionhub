"""Advanced Knowledge Distillation Losses

高级知识蒸馏损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDLoss(nn.Module):
    """Masked Generative Distillation Loss

    掩码生成蒸馏损失
    Paper: Masked Generative Distillation
    """

    def __init__(self, alpha=0.00002, lambda_mgd=0.65):
        super().__init__()
        self.alpha = alpha
        self.lambda_mgd = lambda_mgd

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C, H, W) 学生特征
            teacher_feat: (B, C, H, W) 教师特征
        """
        # 归一化
        student_feat = F.normalize(student_feat, p=2, dim=1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

        # 生成mask
        N, C, H, W = student_feat.shape

        # Spatial attention
        mat = torch.rand((N, 1, H, W)).to(student_feat.device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).float()

        # 计算masked特征距离
        masked_student = student_feat * mat
        masked_teacher = teacher_feat * mat

        loss = F.mse_loss(masked_student, masked_teacher)

        return loss


class DISTLoss(nn.Module):
    """DIST: Knowledge Distillation from A Stronger Teacher

    Paper: Knowledge Distillation from A Stronger Teacher
    """

    def __init__(self, beta=2.0, gamma=2.0, temperature=4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: (B, C) 学生logits
            teacher_logits: (B, C) 教师logits
        """
        T = self.temperature

        # Inter-class correlation
        student_prob = F.softmax(student_logits / T, dim=1)
        teacher_prob = F.softmax(teacher_logits / T, dim=1)

        inter_loss = (teacher_prob - student_prob).pow(2).sum(1).mean()

        # Intra-class correlation
        student_log_prob = F.log_softmax(student_logits / T, dim=1)
        teacher_log_prob = F.log_softmax(teacher_logits / T, dim=1)

        intra_loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction='batchmean'
        )

        loss = self.beta * inter_loss + self.gamma * intra_loss
        loss = loss * (T * T)

        return loss


class AFDLoss(nn.Module):
    """Attention Feature Distillation Loss

    注意力特征蒸馏
    """

    def __init__(self, attention_type='channel'):
        super().__init__()
        self.attention_type = attention_type

    def _channel_attention(self, feat):
        """通道注意力"""
        # Global average pooling
        attention = feat.mean(dim=[2, 3], keepdim=True)
        attention = F.softmax(attention.flatten(1), dim=1)
        return attention.view(feat.size(0), feat.size(1), 1, 1)

    def _spatial_attention(self, feat):
        """空间注意力"""
        # Channel average
        attention = feat.mean(dim=1, keepdim=True)
        attention = F.softmax(attention.flatten(1), dim=1)
        return attention.view(feat.size(0), 1, feat.size(2), feat.size(3))

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C, H, W) 学生特征
            teacher_feat: (B, C, H, W) 教师特征
        """
        if self.attention_type == 'channel':
            student_att = self._channel_attention(student_feat)
            teacher_att = self._channel_attention(teacher_feat)
        else:
            student_att = self._spatial_attention(student_feat)
            teacher_att = self._spatial_attention(teacher_feat)

        # 注意力引导的特征蒸馏
        student_weighted = student_feat * teacher_att
        teacher_weighted = teacher_feat * teacher_att

        loss = F.mse_loss(student_weighted, teacher_weighted)

        # 注意力一致性
        att_loss = F.mse_loss(student_att, teacher_att)

        return loss + 0.1 * att_loss


class DistillationGTCELoss(nn.Module):
    """Ground Truth Cross Entropy Distillation Loss

    真值交叉熵蒸馏损失
    """

    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (B, C) 学生logits
            teacher_logits: (B, C) 教师logits
            labels: (B,) 真值标签
        """
        T = self.temperature

        # 软标签KD损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        # 硬标签CE损失
        hard_loss = self.ce_loss(student_logits, labels)

        # 结合
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss


class DistillationRKDLoss(nn.Module):
    """Relational Knowledge Distillation Loss for Distillation

    蒸馏的关系知识蒸馏损失
    """

    def __init__(self, distance_weight=25.0, angle_weight=50.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, D) 学生特征
            teacher_feat: (B, D) 教师特征
        """
        # Distance-wise loss
        student_dist = self._pdist(student_feat)
        teacher_dist = self._pdist(teacher_feat)

        distance_loss = F.smooth_l1_loss(student_dist, teacher_dist)

        # Angle-wise loss
        student_angle = self._angle(student_feat)
        teacher_angle = self._angle(teacher_feat)

        angle_loss = F.smooth_l1_loss(student_angle, teacher_angle)

        loss = self.distance_weight * distance_loss + self.angle_weight * angle_loss

        return loss

    def _pdist(self, feat):
        """成对距离"""
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=1e-12).sqrt()
        return feat_dist

    def _angle(self, feat):
        """角度"""
        feat_norm = F.normalize(feat, p=2, dim=1)
        angle = torch.mm(feat_norm, feat_norm.t())
        return angle


__all__ = [
    'MGDLoss',
    'DISTLoss',
    'AFDLoss',
    'DistillationGTCELoss',
    'DistillationRKDLoss',
]

