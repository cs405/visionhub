"""ptcls.distill.losses

Teacher-Student 蒸馏常用 loss。

目标：对齐 visionhub 的 KD 思路，但先提供最小可用实现：
- feature/embedding distill: MSE or cosine
- logits distill: KLDiv with temperature
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s, t, reduction="batchmean") * (T * T)


class CosineEmbeddingDistillLoss(nn.Module):
    """1 - cos(student, teacher) 的均值。"""

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student_feat, dim=1)
        t = F.normalize(teacher_feat, dim=1)
        return (1.0 - (s * t).sum(dim=1)).mean()


class MSEFeatLoss(nn.Module):
    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_feat, teacher_feat)
