"""Knowledge Distillation Loss Wrappers

完整的知识蒸馏损失包装器，用于模型蒸馏训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationCELoss(nn.Module):
    """蒸馏交叉熵损失

    用于学生模型学习教师模型的软标签
    """

    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Args:
            student_logits: (B, C) 学生模型输出
            teacher_logits: (B, C) 教师模型输出
            labels: (B,) 真实标签（可选）
        """
        T = self.temperature

        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        if labels is not None:
            # 硬标签损失
            hard_loss = self.ce_loss(student_logits, labels)
            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return soft_loss


class DistillationKLDivLoss(nn.Module):
    """蒸馏KL散度损失"""

    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        return loss


class DistillationDMLLoss(nn.Module):
    """蒸馏深度互学习损失"""

    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits1, logits2):
        T = self.temperature

        # 双向KL散度
        loss1 = F.kl_div(
            F.log_softmax(logits1 / T, dim=1),
            F.softmax(logits2 / T, dim=1),
            reduction='batchmean'
        )

        loss2 = F.kl_div(
            F.log_softmax(logits2 / T, dim=1),
            F.softmax(logits1 / T, dim=1),
            reduction='batchmean'
        )

        return (loss1 + loss2) / 2 * (T * T)


class DistillationDistanceLoss(nn.Module):
    """蒸馏距离损失"""

    def __init__(self, mode='l2'):
        super().__init__()
        self.mode = mode

        if mode == 'l1':
            self.loss_fn = nn.L1Loss()
        elif mode == 'l2':
            self.loss_fn = nn.MSELoss()
        elif mode == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, student_features, teacher_features):
        return self.loss_fn(student_features, teacher_features)


class MultiLabelAsymmetricLoss(nn.Module):
    """多标签非对称损失

    Paper: Asymmetric Loss For Multi-Label Classification
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) 模型输出
            targets: (B, C) 多标签目标
        """
        # Sigmoid
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            loss = -(los_pos + los_neg)

        return loss.mean()


__all__ = [
    'DistillationCELoss',
    'DistillationKLDivLoss',
    'DistillationDMLLoss',
    'DistillationDistanceLoss',
    'MultiLabelAsymmetricLoss',
]

