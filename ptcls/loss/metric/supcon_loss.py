"""Supervised Contrastive Learning Loss

Implementation of supervised contrastive learning
Paper: Supervised Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss

    Paper: Supervised Contrastive Learning
    Reference: https://arxiv.org/pdf/2004.11362.pdf
    Code: https://github.com/HobbitLong/SupContrast

    也支持SimCLR中的无监督对比学习损失

    Args:
        views: 视图数量，默认16
        temperature: 温度参数，默认0.07
        contrast_mode: 对比模式 ('one' 或 'all')，默认'all'
        base_temperature: 基础温度，默认0.07
        normalize_feature: 是否归一化特征，默认True
    """

    def __init__(
        self,
        views=16,
        temperature=0.07,
        contrast_mode='all',
        base_temperature=0.07,
        normalize_feature=True
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.num_ids = None
        self.views = views
        self.normalize_feature = normalize_feature

    def forward(self, features, labels, mask=None):
        """计算监督对比学习损失

        Args:
            features: 特征向量 [bsz * n_views, ...] 或 [bsz, n_views, ...]
            labels: 标签 [bsz] 或 [bsz * n_views]
            mask: 对比掩码 [bsz, bsz]，可选

        Returns:
            loss: 标量损失值
        """
        device = features.device

        # 处理输入维度
        if features.dim() == 2:
            # [bsz * n_views, D] -> [bsz, n_views, D]
            if self.num_ids is None:
                self.num_ids = features.size(0) // self.views
            features = features.view(self.num_ids, self.views, -1)

        # 归一化
        if self.normalize_feature:
            features = F.normalize(features, p=2, dim=-1)

        # 处理标签
        if labels.dim() == 1 and labels.size(0) == features.size(0) * features.size(1):
            # [bsz * n_views] -> [bsz]
            labels = labels.view(features.size(0), features.size(1))[:, 0]
        elif labels.dim() == 2:
            labels = labels[:, 0]

        batch_size = features.size(0)

        # 确保至少3维
        if features.dim() < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions required')

        if features.dim() > 3:
            features = features.view(features.size(0), features.size(1), -1)

        # 创建mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.size(0) != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()

        contrast_count = features.size(1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # 掩码去除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算正样本的平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


__all__ = ['SupConLoss']

