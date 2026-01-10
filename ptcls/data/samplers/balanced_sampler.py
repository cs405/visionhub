"""BalancedBatchSampler: Balanced Sampling for Imbalanced Datasets

平衡批次采样器，用于处理类别不平衡的数据集
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class BalancedBatchSampler(Sampler):
    """Balanced Batch Sampler

    确保每个batch中各类别样本数量相对均衡

    Args:
        dataset: 数据集
        batch_size: 批次大小
        samples_per_class: 每个类别每批次的样本数
        drop_last: 是否丢弃最后不完整的batch
    """

    def __init__(self, dataset, batch_size, samples_per_class=None, drop_last=True):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 获取标签
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            raise ValueError("Dataset must have 'labels' or 'targets' attribute")

        # 构建类别到索引的映射
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        # 计算每个类别应该采样多少样本
        if samples_per_class is None:
            self.samples_per_class = max(1, batch_size // self.num_classes)
        else:
            self.samples_per_class = samples_per_class

        # 计算总长度
        self.num_samples = len(dataset)
        if drop_last:
            self.num_batches = self.num_samples // batch_size
        else:
            self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        # 为每个类别创建随机顺序
        class_iters = {}
        for cls in self.classes:
            indices = self.class_to_indices[cls].copy()
            np.random.shuffle(indices)
            class_iters[cls] = iter(indices)

        # 生成批次
        for _ in range(self.num_batches):
            batch = []

            # 从每个类别采样
            for cls in self.classes:
                samples_needed = min(self.samples_per_class, self.batch_size - len(batch))

                for _ in range(samples_needed):
                    try:
                        idx = next(class_iters[cls])
                        batch.append(idx)
                    except StopIteration:
                        # 如果该类别样本用完，重新shuffle
                        indices = self.class_to_indices[cls].copy()
                        np.random.shuffle(indices)
                        class_iters[cls] = iter(indices)
                        try:
                            idx = next(class_iters[cls])
                            batch.append(idx)
                        except StopIteration:
                            break

                if len(batch) >= self.batch_size:
                    break

            # 如果batch不够大，随机补充
            while len(batch) < self.batch_size:
                cls = np.random.choice(self.classes)
                try:
                    idx = next(class_iters[cls])
                    batch.append(idx)
                except StopIteration:
                    indices = self.class_to_indices[cls].copy()
                    np.random.shuffle(indices)
                    class_iters[cls] = iter(indices)
                    idx = next(class_iters[cls])
                    batch.append(idx)

            # 随机打乱batch内顺序
            np.random.shuffle(batch)

            yield from batch[:self.batch_size]

    def __len__(self):
        if self.drop_last:
            return self.num_batches * self.batch_size
        else:
            return self.num_samples


class DomainShuffleSampler(Sampler):
    """Domain Shuffle Sampler

    域混洗采样器，用于多域学习

    Args:
        dataset: 数据集，必须有domain_labels属性
        batch_size: 批次大小
        shuffle_domains: 是否混洗域
    """

    def __init__(self, dataset, batch_size, shuffle_domains=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_domains = shuffle_domains

        if not hasattr(dataset, 'domain_labels'):
            raise ValueError("Dataset must have 'domain_labels' attribute")

        # 按域分组
        self.domain_to_indices = defaultdict(list)
        for idx, domain in enumerate(dataset.domain_labels):
            self.domain_to_indices[domain].append(idx)

        self.domains = list(self.domain_to_indices.keys())
        self.num_samples = len(dataset)

    def __iter__(self):
        indices = []

        if self.shuffle_domains:
            # 打乱域的顺序
            domains = self.domains.copy()
            np.random.shuffle(domains)
        else:
            domains = self.domains

        # 对每个域内的样本打乱
        for domain in domains:
            domain_indices = self.domain_to_indices[domain].copy()
            np.random.shuffle(domain_indices)
            indices.extend(domain_indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


__all__ = [
    'BalancedBatchSampler',
    'DomainShuffleSampler',
]

