"""ptcls.loss.arcface

ArcFace head (Additive Angular Margin Softmax) for metric learning.

We implement a minimal ArcFace classification head:
- Input features: [B, D] (embeddings)
- Output logits: [B, num_classes]

Usage:
- During training, compute ArcFace logits and use cross-entropy.
- During inference for retrieval, you usually discard classifier and use embeddings.

This design keeps the retrieval embedding head separate from classifier.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = embeddings @ W.t()  # [B,C]
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return logits

