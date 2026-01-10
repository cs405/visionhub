"""MetaBinSampler for Meta-learning Binary Networks

元学习二值化网络的采样器
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class MetaBinSampler(Sampler):
    """MetaBIN采样器

    用于元学习二值化网络，确保每个batch包含多个任务

    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_tasks: 每批次的任务数
        num_samples_per_task: 每个任务的样本数
        shuffle: 是否打乱
    """

    def __init__(self, dataset, batch_size, num_tasks=4, num_samples_per_task=8, shuffle=True):
        super().__init__(data_source=None)

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_samples_per_task = num_samples_per_task
        self.shuffle = shuffle

        # 获取标签
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            raise ValueError("Dataset must have 'labels' or 'targets' attribute")

        # 按类别分组
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        # 验证参数
        assert batch_size >= num_tasks * num_samples_per_task, \
            f"batch_size ({batch_size}) must >= num_tasks ({num_tasks}) * num_samples_per_task ({num_samples_per_task})"

    def __iter__(self):
        """生成批次索引"""
        # 计算批次数
        num_batches = len(self.dataset) // self.batch_size

        indices = []

        for _ in range(num_batches):
            batch_indices = []

            # 随机选择num_tasks个类别作为任务
            if self.shuffle:
                selected_classes = np.random.choice(
                    self.classes,
                    size=min(self.num_tasks, self.num_classes),
                    replace=False
                )
            else:
                selected_classes = self.classes[:self.num_tasks]

            # 为每个任务采样样本
            for cls in selected_classes:
                class_indices = self.class_to_indices[cls]

                # 采样样本
                if len(class_indices) >= self.num_samples_per_task:
                    sampled = np.random.choice(
                        class_indices,
                        size=self.num_samples_per_task,
                        replace=False
                    )
                else:
                    # 不足时重复采样
                    sampled = np.random.choice(
                        class_indices,
                        size=self.num_samples_per_task,
                        replace=True
                    )

                batch_indices.extend(sampled.tolist())

            # 如果batch不够大，随机补充
            while len(batch_indices) < self.batch_size:
                cls = np.random.choice(self.classes)
                idx = np.random.choice(self.class_to_indices[cls])
                batch_indices.append(idx)

            # 截断到batch_size
            batch_indices = batch_indices[:self.batch_size]

            if self.shuffle:
                np.random.shuffle(batch_indices)

            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self):
        return (len(self.dataset) // self.batch_size) * self.batch_size


__all__ = ['MetaBinSampler']

