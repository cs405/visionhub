"""Knowledge Distillation Loss Functions

Includes: KLDivLoss, DKDLoss, RKDLoss, SKDLoss, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ KLDivLoss (Knowledge Distillation) ============
class KLDivLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Args:
            student_logits: (B, C)
            teacher_logits: (B, C)
            labels: (B,) optional, for combining with CE loss
        """
        T = self.temperature

        # Soft targets
        soft_log_probs = F.log_softmax(student_logits / T, dim=1)
        soft_targets = F.softmax(teacher_logits / T, dim=1)

        kd_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (T * T)

        if labels is not None:
            # Combine with hard target loss
            ce_loss = F.cross_entropy(student_logits, labels)
            loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
            return loss

        return kd_loss


# ============ DKDLoss (Decoupled Knowledge Distillation) ============
class DKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (B, C)
            teacher_logits: (B, C)
            labels: (B,)
        """
        T = self.temperature
        batch_size, num_classes = student_logits.shape

        # Get teacher and student predictions
        student_probs = F.softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)

        # TCKD: Target Class Knowledge Distillation
        student_target = student_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        teacher_target = teacher_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        tckd_loss = F.binary_cross_entropy(student_target, teacher_target)

        # NCKD: Non-target Class Knowledge Distillation
        student_probs_copy = student_probs.clone()
        teacher_probs_copy = teacher_probs.clone()
        student_probs_copy.scatter_(1, labels.unsqueeze(1), 0)
        teacher_probs_copy.scatter_(1, labels.unsqueeze(1), 0)

        student_probs_copy = student_probs_copy / (1 - student_target.unsqueeze(1) + 1e-7)
        teacher_probs_copy = teacher_probs_copy / (1 - teacher_target.unsqueeze(1) + 1e-7)

        nckd_loss = F.kl_div(
            torch.log(student_probs_copy + 1e-7),
            teacher_probs_copy,
            reduction='batchmean'
        )

        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        loss = loss * (T * T)

        return loss


# ============ RKDLoss (Relational Knowledge Distillation) ============
class RKDLoss(nn.Module):
    def __init__(self, w_dist=25.0, w_angle=50.0):
        super().__init__()
        self.w_dist = w_dist
        self.w_angle = w_angle

    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        # Distance loss
        with torch.no_grad():
            t_dist = self.pdist(teacher_features, squared=False)
            mean_td = t_dist[t_dist > 0].mean()
            t_dist = t_dist / mean_td

        d_dist = self.pdist(student_features, squared=False)
        mean_d = d_dist[d_dist > 0].mean()
        d_dist = d_dist / mean_d

        dist_loss = F.smooth_l1_loss(d_dist, t_dist)

        # Angle loss
        with torch.no_grad():
            td = (teacher_features.unsqueeze(0) - teacher_features.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student_features.unsqueeze(0) - student_features.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        angle_loss = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_dist * dist_loss + self.w_angle * angle_loss

        return loss


# ============ SKDLoss (Self Knowledge Distillation) ============
class SKDLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits_list):
        """
        Args:
            student_logits_list: List[(B, C)] multiple outputs from student
        """
        if len(student_logits_list) < 2:
            return torch.tensor(0.0, device=student_logits_list[0].device)

        T = self.temperature
        total_loss = 0.0
        count = 0

        for i in range(len(student_logits_list)):
            for j in range(i + 1, len(student_logits_list)):
                logits_i = student_logits_list[i]
                logits_j = student_logits_list[j]

                soft_log_probs_i = F.log_softmax(logits_i / T, dim=1)
                soft_targets_j = F.softmax(logits_j / T, dim=1)

                loss = F.kl_div(soft_log_probs_i, soft_targets_j, reduction='batchmean')
                total_loss += loss
                count += 1

        return (total_loss / count) * (T * T) if count > 0 else torch.tensor(0.0)


# ============ AttentionTransferLoss (AT Loss) ============
class AttentionTransferLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def attention_map(self, fm):
        """
        Args:
            fm: (B, C, H, W)
        Returns:
            attention: (B, H, W)
        """
        return torch.pow(fm.abs(), self.p).sum(dim=1)

    def forward(self, student_feature_maps, teacher_feature_maps):
        """
        Args:
            student_feature_maps: List[(B, C, H, W)]
            teacher_feature_maps: List[(B, C, H, W)]
        """
        total_loss = 0.0

        for s_fm, t_fm in zip(student_feature_maps, teacher_feature_maps):
            # Resize if needed
            if s_fm.shape[-2:] != t_fm.shape[-2:]:
                s_fm = F.interpolate(s_fm, size=t_fm.shape[-2:], mode='bilinear', align_corners=False)

            s_attn = self.attention_map(s_fm)
            t_attn = self.attention_map(t_fm)

            # Normalize
            s_attn = F.normalize(s_attn.view(s_attn.size(0), -1))
            t_attn = F.normalize(t_attn.view(t_attn.size(0), -1))

            loss = (s_attn - t_attn).pow(2).mean()
            total_loss += loss

        return total_loss / len(student_feature_maps)


# ============ FactorTransferLoss (FT Loss) ============
class FactorTransferLoss(nn.Module):
    def __init__(self, beta=250.0):
        super().__init__()
        self.beta = beta

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        # Normalize
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Factor Transfer: preserve teacher's covariance
        teacher_cov = teacher_norm.t() @ teacher_norm
        student_cov = student_norm.t() @ student_norm

        loss = (teacher_cov - student_cov).pow(2).sum()

        return loss / (student_features.size(0) ** 2) * self.beta


# ============ SimilarityPreservingLoss (SP Loss) ============
class SimilarityPreservingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: (B, D)
            teacher_features: (B, D)
        """
        # Normalize
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)

        # Similarity matrices
        student_sim = student_norm @ student_norm.t()
        teacher_sim = teacher_norm @ teacher_norm.t()

        loss = F.mse_loss(student_sim, teacher_sim)

        return loss


# Export all
__all__ = [
    'KLDivLoss',
    'DKDLoss',
    'RKDLoss',
    'SKDLoss',
    'AttentionTransferLoss',
    'FactorTransferLoss',
    'SimilarityPreservingLoss',
]

