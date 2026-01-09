"""ptcls.arch.head.embedding_head

一个最小但实用的 embedding head：
- Linear -> BN1d -> (optional) ReLU

用于把 backbone 的 feature 向量映射到可配置 embedding_size。

说明：
- 当前 ResNet backbone 的 forward 返回的是分类 logits（fc 输出）。要做正经 embedding，
  更理想的是从 fc 前的 pooled feature（通常 512/2048）接 head。
- 为了尽量少改动并让 pipeline 先可用，这里支持两种输入：
  1) 输入就是 feature（推荐，后续我们会改 ResNet 输出 feature）
  2) 输入是 logits（临时兼容）

后续增强：可改成 MLP、加 dropout、或者支持 ArcFace/CosFace。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """Embedding head for retrieval.

    Default behavior (recommended, like many ReID setups):
    - Linear projection to embedding_size
    - BNNeck (BatchNorm1d)
    - Optional (during inference) L2 normalize

    Notes:
    - For metric learning losses (Triplet/SupCon), using the BN output is common.
    - We keep this head minimal and stable.
    """

    def __init__(
        self,
        in_dim: int,
        embedding_size: int = 512,
        with_relu: bool = False,
        with_l2norm: bool = True,
        bn_affine: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, affine=bn_affine)
        self.with_relu = bool(with_relu)
        self.with_l2norm = bool(with_l2norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        if self.with_relu:
            x = F.relu(x)
        if self.with_l2norm:
            x = F.normalize(x, dim=1)
        return x

