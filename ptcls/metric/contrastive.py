"""ptcls.metric.contrastive

一些常用的 metric-learning loss（Professional）。

后续可以逐步对齐 visionhub 的 loss/metric 实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).

    输入：
    - features: [B, D] (已归一化更稳定)
    - labels: [B]

    说明：这是一个简化实现，先服务于“训练 embedding head 提升检索效果”的最小闭环。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must be [B, D]")

        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        feats = F.normalize(features, dim=1)
        logits = torch.matmul(feats, feats.T) / self.temperature

        # remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        # log_prob
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

