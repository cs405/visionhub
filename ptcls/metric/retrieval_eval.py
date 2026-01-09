"""ptcls.metric.retrieval_eval

检索评估：Recall@K / mAP@K（基于 embedding + label）

这是一个最小但实用的评估实现：
- 给定 query embeddings/labels 和 gallery embeddings/labels
- 计算 cosine similarity

说明：
- 对齐 visionhub 思路：训练完直接输出 Recall@1/5/10 和 mAP@10
- 后续可以扩展为：支持多标签、支持排除同图、支持分组等。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def recall_at_k(sim: np.ndarray, q_labels: Sequence[int], g_labels: Sequence[int], ks=(1, 5, 10)) -> Dict[int, float]:
    q_labels = np.asarray(q_labels)
    g_labels = np.asarray(g_labels)
    order = np.argsort(-sim, axis=1)  # desc

    res: Dict[int, float] = {}
    for k in ks:
        topk = order[:, :k]
        hit = (g_labels[topk] == q_labels[:, None]).any(axis=1)
        res[int(k)] = float(hit.mean())
    return res


def map_at_k(sim: np.ndarray, q_labels: Sequence[int], g_labels: Sequence[int], k: int = 10) -> float:
    q_labels = np.asarray(q_labels)
    g_labels = np.asarray(g_labels)

    # k=None means use full gallery (mAP@all)
    if k is None:
        k_eff = int(sim.shape[1])
    else:
        k_eff = int(k)
        k_eff = max(1, min(k_eff, int(sim.shape[1])))

    order = np.argsort(-sim, axis=1)[:, :k_eff]

    aps: List[float] = []
    for i in range(order.shape[0]):
        ranked = order[i]
        rel = (g_labels[ranked] == q_labels[i]).astype(np.int32)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        prec = np.cumsum(rel) / (np.arange(k_eff) + 1)
        ap = float((prec * rel).sum() / rel.sum())
        aps.append(ap)
    return float(np.mean(aps))
