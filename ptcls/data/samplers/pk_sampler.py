"""ptcls.data.samplers.pk_sampler

PKSampler（P 类 * 每类 K 个样本）
- 对检索/度量学习训练非常关键
- 一个 batch 固定包含 P 个类别，每个类别采样 K 张图

用法（示例）：
- 传入 labels（长度=dataset size，元素为 int class_id）
- DataLoader(..., batch_sampler=PKBatchSampler(...))

注意：
- 这是 batch_sampler（直接 yield list[int] indices），不要再额外传 batch_size/shuffle。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, Iterator, List, Sequence


class PKBatchSampler:
    def __init__(self, labels: Sequence[int], P: int, K: int, drop_last: bool = True, seed: int = 0):
        if P <= 0 or K <= 0:
            raise ValueError("P and K must be > 0")
        self.labels = list(map(int, labels))
        self.P = int(P)
        self.K = int(K)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(seed)

        self.class_to_indices: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        if len(self.classes) < self.P:
            raise ValueError(f"Not enough classes for P={self.P}. Found classes={len(self.classes)}")

    def __iter__(self) -> Iterator[List[int]]:
        """Yield a finite number of batches per epoch.

        IMPORTANT:
        - DataLoader expects the batch_sampler iterator to end; otherwise an epoch will never finish.
        - We generate exactly `len(self)` batches.
        """
        num_batches = len(self)
        # shuffle class order each epoch for better coverage
        classes = list(self.classes)
        self.rng.shuffle(classes)

        for _ in range(num_batches):
            chosen_classes = self.rng.sample(self.classes, self.P)
            batch: List[int] = []

            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.K:
                    batch.extend(self.rng.sample(idxs, self.K))
                else:
                    # not enough samples: sample with replacement
                    batch.extend([self.rng.choice(idxs) for _ in range(self.K)])

            # Safety: always yield exactly P*K indices
            if len(batch) != self.P * self.K:
                if self.drop_last:
                    continue
                # pad (rare)
                if len(batch) < self.P * self.K and len(batch) > 0:
                    batch.extend([batch[-1]] * (self.P * self.K - len(batch)))
                batch = batch[: self.P * self.K]

            yield batch

    def __len__(self) -> int:
        # 这是一个近似值：按“每个 batch 用掉 P*K 个样本”估算
        n = len(self.labels)
        b = self.P * self.K
        return n // b if self.drop_last else (n + b - 1) // b

