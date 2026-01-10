"""MetaBIN and Advanced Losses

MetaBIN元学习和其他高级损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CELossForMetaBIN(nn.Module):
    """MetaBIN的交叉熵损失

    用于元学习二值化网络
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C) 模型输出
            labels: (B,) 标签
        """
        return self.ce_loss(logits, labels)


class TripletLossForMetaBIN(nn.Module):
    """MetaBIN的三元组损失"""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 特征嵌入
            labels: (B,) 标签
        """
        # 计算距离矩阵
        dist = torch.cdist(embeddings, embeddings, p=2)

        # 构建正负样本mask
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()
        mask_pos.fill_diagonal_(0)

        # 困难正样本
        dist_ap = (dist * mask_pos + (1 - mask_pos) * 1e9).min(dim=1)[0]

        # 困难负样本
        dist_an = (dist * mask_neg + (1 - mask_neg) * 1e9).min(dim=1)[0]

        # Triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin).mean()

        return loss


class InterDomainShuffleLoss(nn.Module):
    """域间混洗损失

    用于多域学习，鼓励不同域之间的特征混洗
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, domain_labels):
        """
        Args:
            features: (B, D) 特征
            domain_labels: (B,) 域标签
        """
        batch_size = features.size(0)

        # 归一化
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 构建域间mask
        domain_labels = domain_labels.unsqueeze(1)
        inter_domain_mask = (domain_labels != domain_labels.T).float()

        # 对比损失
        exp_sim = torch.exp(sim_matrix)

        # 分母：所有样本
        denominator = exp_sim.sum(dim=1, keepdim=True)

        # 分子：不同域的样本
        numerator = (exp_sim * inter_domain_mask).sum(dim=1, keepdim=True)

        # 损失
        loss = -torch.log(numerator / (denominator + 1e-8)).mean()

        return loss


class IntraDomainScatterLoss(nn.Module):
    """域内散布损失

    鼓励同一域内的样本分散分布
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, domain_labels):
        """
        Args:
            features: (B, D) 特征
            domain_labels: (B,) 域标签
        """
        # 按域分组
        unique_domains = torch.unique(domain_labels)

        loss = 0
        count = 0

        for domain in unique_domains:
            mask = (domain_labels == domain)
            domain_features = features[mask]

            if domain_features.size(0) > 1:
                # 计算域内距离
                dist = torch.cdist(domain_features, domain_features, p=2)

                # 鼓励距离大于margin
                loss += F.relu(self.margin - dist).mean()
                count += 1

        if count > 0:
            loss = loss / count

        return loss


class PEFDLoss(nn.Module):
    """Patch-Enhanced Feature Distillation Loss

    补丁增强特征蒸馏损失
    """

    def __init__(self, patch_size=7, alpha=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = alpha

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C, H, W) 学生特征
            teacher_feat: (B, C, H, W) 教师特征
        """
        B, C, H, W = student_feat.shape

        # 全局特征蒸馏
        global_loss = F.mse_loss(student_feat, teacher_feat)

        # Patch特征蒸馏
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        student_patches = student_feat.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        teacher_patches = teacher_feat.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

        # 计算每个patch的重要性
        teacher_importance = teacher_patches.pow(2).mean(dim=(1, 4, 5), keepdim=True)
        teacher_importance = F.softmax(teacher_importance.flatten(2), dim=2).reshape_as(teacher_importance)

        # 加权patch损失
        patch_loss = ((student_patches - teacher_patches).pow(2) * teacher_importance).mean()

        return global_loss + self.alpha * patch_loss


class CCSSLCELoss(nn.Module):
    """Cross-modal Contrastive Self-Supervised Learning CE Loss

    跨模态对比自监督学习交叉熵损失
    """

    def __init__(self, temperature=0.07, lambda_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features1, features2, labels):
        """
        Args:
            features1: (B, D) 模态1特征
            features2: (B, D) 模态2特征
            labels: (B,) 标签
        """
        # 归一化
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        # 跨模态相似度
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature

        # 对比学习目标：对角线为正样本
        targets = torch.arange(features1.size(0)).to(features1.device)

        # 双向对比损失
        loss_12 = self.ce_loss(sim_matrix, targets)
        loss_21 = self.ce_loss(sim_matrix.T, targets)

        contrastive_loss = (loss_12 + loss_21) / 2

        return contrastive_loss


__all__ = [
    'CELossForMetaBIN',
    'TripletLossForMetaBIN',
    'InterDomainShuffleLoss',
    'IntraDomainScatterLoss',
    'PEFDLoss',
    'CCSSLCELoss',
]

