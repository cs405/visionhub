"""ptcls.metric.triplet

Batch-hard Triplet Loss（基于 cosine 距离）。

适用于 PK 采样（P 类 * K 样本）训练：
- hardest positive: 同类中距离最大的样本
- hardest negative: 异类中距离最小的样本

输入：
- embeddings: [B, D]
- labels: [B]

输出：
- scalar loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [B, D]")

        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        B = embeddings.size(0)

        # cosine distance: 1 - cos
        sim = embeddings @ embeddings.t()
        dist = 1.0 - sim

        # masks
        is_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        is_neg = ~is_pos

        # remove self from positive mask
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        is_pos = is_pos & ~eye

        # hardest positive (max dist among positives)
        pos_dist = dist.masked_fill(~is_pos, float("-inf"))
        hardest_pos, _ = pos_dist.max(dim=1)

        # hardest negative (min dist among negatives)
        neg_dist = dist.masked_fill(~is_neg, float("inf"))
        hardest_neg, _ = neg_dist.min(dim=1)

        valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
        if not valid.any():
            return embeddings.new_tensor(0.0)

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()
