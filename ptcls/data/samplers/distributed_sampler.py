"""DistributedRandomIdentitySampler

分布式随机身份采样器，用于ReID任务
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class DistributedRandomIdentitySampler(Sampler):
    """分布式随机身份采样器

    用于行人重识别等任务，确保每个batch包含多个身份，每个身份有多个样本

    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_instances: 每个身份的实例数
        rank: 当前进程rank
        world_size: 总进程数
        seed: 随机种子
    """

    def __init__(self, dataset, batch_size, num_instances=4, rank=0, world_size=1, seed=0):
        super().__init__(data_source=None)

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        # 获取标签
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'pids'):
            labels = dataset.pids
        else:
            raise ValueError("Dataset must have 'labels', 'targets' or 'pids' attribute")

        # 构建身份到索引的映射
        self.index_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            self.index_dict[label].append(idx)

        self.pids = list(self.index_dict.keys())

        # 计算每个身份需要的批次数
        self.num_identities = len(self.pids)
        self.num_pids_per_batch = batch_size // num_instances

        # 计算总长度
        self.num_samples = len(labels)

        # 为每个进程计算样本数
        self.num_samples_per_rank = int(np.ceil(self.num_samples / world_size))
        self.total_size = self.num_samples_per_rank * world_size

        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.seed + self.epoch)

        # 为每个身份生成随机索引
        final_idxs = []

        # 随机打乱身份顺序
        pids = np.random.permutation(self.pids).tolist()

        # 为每个身份采样实例
        for pid in pids:
            idxs = self.index_dict[pid]

            if len(idxs) < self.num_instances:
                # 如果样本不足，重复采样
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                # 随机采样
                idxs = np.random.choice(idxs, size=self.num_instances, replace=False)

            final_idxs.extend(idxs)

        # 填充到total_size
        if len(final_idxs) < self.total_size:
            extra = np.random.choice(
                range(len(self.dataset)),
                size=self.total_size - len(final_idxs),
                replace=True
            )
            final_idxs.extend(extra.tolist())

        # 截断到total_size
        final_idxs = final_idxs[:self.total_size]

        # 分配给当前进程
        offset = self.num_samples_per_rank * self.rank
        final_idxs = final_idxs[offset:offset + self.num_samples_per_rank]

        return iter(final_idxs)

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        """设置epoch用于改变随机性"""
        self.epoch = epoch


class DistributedGivenIterationSampler(Sampler):
    """分布式固定迭代次数采样器

    Args:
        dataset: 数据集
        total_iter: 总迭代次数
        batch_size: 批次大小
        rank: 当前进程rank
        world_size: 总进程数
        last_iter: 最后一次迭代
        seed: 随机种子
    """

    def __init__(self, dataset, total_iter, batch_size, rank=0, world_size=1, last_iter=-1, seed=0):
        super().__init__(data_source=None)

        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.last_iter = last_iter
        self.seed = seed

        self.total_size = total_iter * batch_size * world_size
        self.num_samples = self.total_size // world_size

        self.call = 0

    def __iter__(self):
        # 确定性shuffle
        if self.last_iter >= self.total_iter:
            self.last_iter = -1

        np.random.seed(self.seed)

        indices = []
        # 生成足够的索引
        num_rounds = int(np.ceil(self.total_size / len(self.dataset)))

        for _ in range(num_rounds):
            shuffled = np.random.permutation(len(self.dataset)).tolist()
            indices.extend(shuffled)

        # 截断到total_size
        indices = indices[:self.total_size]

        # 分配给当前进程
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]

        # 跳过已完成的迭代
        start = (self.last_iter + 1) * self.batch_size
        indices = indices[start:]

        self.call += 1

        return iter(indices)

    def __len__(self):
        # 剩余样本数
        return self.num_samples - (self.last_iter + 1) * self.batch_size


__all__ = [
    'DistributedRandomIdentitySampler',
    'DistributedGivenIterationSampler',
]

