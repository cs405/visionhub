"""Deep Hash Learning Losses

深度哈希学习损失函数
"""

import torch
import torch.nn as nn


class DSHSDLoss(nn.Module):
    """Deep Supervised Hashing with Pairwise Labels

    Paper: Deep Supervised Hashing with Pairwise Labels
    """

    def __init__(self, alpha=0.01, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, u, y):
        """
        Args:
            u: (B, hash_bits) hash codes
            y: (B,) labels
        """
        batch_size = u.size(0)

        # 构建相似度矩阵
        y = y.view(-1, 1)
        s = (y == y.T).float()

        # 内积
        inner_product = u @ u.T / 2

        # 对数似然损失
        likelihood_loss = (torch.log(1 + torch.exp(inner_product)) - s * inner_product).mean()

        # 量化损失
        quantization_loss = (u.abs() - 1).pow(2).mean()

        loss = likelihood_loss + self.alpha * quantization_loss
        return loss


class DCHLoss(nn.Module):
    """Deep Cauchy Hashing Loss

    Paper: Deep Cauchy Hashing for Hamming Space Retrieval
    """

    def __init__(self, gamma=20, lambda_=0.0001):
        super().__init__()
        self.gamma = gamma
        self.lambda_ = lambda_

    def forward(self, u, y):
        """
        Args:
            u: (B, hash_bits) hash codes
            y: (B,) labels
        """
        batch_size = u.size(0)
        hash_bits = u.size(1)

        # 构建相似度标签
        y = y.view(-1, 1)
        s = (y == y.T).float() * 2 - 1  # {-1, 1}

        # Hamming距离
        dist = 0.5 * (hash_bits - u @ u.T)

        # Cauchy分布
        cauchy = self.gamma / (dist + self.gamma)

        # 对比损失
        s_mask = (s == 1)
        d_mask = (s == -1)

        loss_positive = (dist[s_mask]).mean() if s_mask.any() else 0
        loss_negative = (torch.log(1 + cauchy[d_mask])).mean() if d_mask.any() else 0

        # 正则化
        regularization = (u.abs() - 1).pow(2).mean()

        loss = loss_positive + loss_negative + self.lambda_ * regularization
        return loss


class LCDSHLoss(nn.Module):
    """Label Consistent Deep Supervised Hashing Loss

    标签一致性深度监督哈希
    """

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, u, y, labels_onehot=None):
        """
        Args:
            u: (B, hash_bits) hash codes
            y: (B,) labels
            labels_onehot: (B, num_classes) one-hot labels (optional)
        """
        batch_size = u.size(0)
        hash_bits = u.size(1)

        # 构建相似度矩阵
        y = y.view(-1, 1)
        s = (y == y.T).float()

        # 内积相似度
        theta = u @ u.T / 2

        # 似然损失
        likelihood_loss = -torch.mean(s * theta - torch.log(1 + torch.exp(theta)))

        # 量化损失
        quantization_loss = torch.mean((u.abs() - 1).pow(2))

        loss = likelihood_loss + self.alpha * quantization_loss

        # 如果提供了one-hot标签，添加标签平衡损失
        if labels_onehot is not None:
            # 确保每个bit能够区分不同类别
            codes_mean = []
            for i in range(labels_onehot.size(1)):
                mask = labels_onehot[:, i] == 1
                if mask.any():
                    codes_mean.append(u[mask].mean(0))

            if len(codes_mean) > 1:
                codes_mean = torch.stack(codes_mean)
                # 不同类别的hash code应该不同
                balance_loss = -torch.pdist(codes_mean, p=2).mean()
                loss = loss + self.beta * balance_loss

        return loss


__all__ = [
    'DSHSDLoss',
    'DCHLoss',
    'LCDSHLoss',
]

