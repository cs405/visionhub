"""ptcls.loss.circle_loss

Circle Loss (classification-free metric learning).

Reference idea:
- Encourage positive pairs to have high similarity
- Encourage negative pairs to have low similarity

This implementation follows a common supervised CircleLoss formulation:
- Input embeddings: [B, D]
- Labels: [B]
- Uses cosine similarity matrix

Practical notes:
- Works best with PK sampling.
- Typically use normalized embeddings.

This is a minimal, practical implementation for retrieval.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, m: float = 0.25, gamma: float = 256.0):
        super().__init__()
        self.m = float(m)
        self.gamma = float(gamma)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [B, D]")
        labels = labels.view(-1)
        B = embeddings.size(0)
        if labels.numel() != B:
            raise ValueError("labels must be [B]")

        # cosine sim
        sim = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).t()

        # pos/neg masks
        is_pos = labels[:, None].eq(labels[None, :])
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        is_pos = is_pos & ~eye
        is_neg = ~labels[:, None].eq(labels[None, :])

        # gather pos/neg similarities
        sp = sim[is_pos]
        sn = sim[is_neg]

        if sp.numel() == 0 or sn.numel() == 0:
            return embeddings.new_tensor(0.0)

        # CircleLoss weighting
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, 0.0)
        an = torch.clamp_min(sn.detach() + self.m, 0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)

        # Use logsumexp for stability
        loss_p = torch.logsumexp(logit_p, dim=0)
        loss_n = torch.logsumexp(logit_n, dim=0)

        loss = F.softplus(loss_p + loss_n)
        return loss

