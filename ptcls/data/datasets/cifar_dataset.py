"""CIFAR Dataset Wrappers

CIFAR-10和CIFAR-100数据集包装器，用于PaddleClas风格的训练
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pickle


class CIFARDataset(Dataset):
    """CIFAR数据集包装器

    支持CIFAR-10和CIFAR-100

    Args:
        root: 数据根目录
        train: 是否为训练集
        transform: 数据变换
        cifar100: 是否为CIFAR-100（默认CIFAR-10）
    """

    def __init__(self, root, train=True, transform=None, cifar100=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.cifar100 = cifar100

        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """加载CIFAR数据"""
        if self.cifar100:
            # CIFAR-100
            if self.train:
                file_list = ['train']
            else:
                file_list = ['test']
        else:
            # CIFAR-10
            if self.train:
                file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                           'data_batch_4', 'data_batch_5']
            else:
                file_list = ['test_batch']

        for filename in file_list:
            filepath = os.path.join(self.root, filename)

            with open(filepath, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')

                self.data.append(entry[b'data'])

                if b'labels' in entry:
                    self.labels.extend(entry[b'labels'])
                else:
                    self.labels.extend(entry[b'fine_labels'])

        # 合并数据
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC

    @property
    def targets(self):
        """兼容性属性"""
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # 转换为PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class CIFAR10Dataset(CIFARDataset):
    """CIFAR-10数据集"""
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform, cifar100=False)


class CIFAR100Dataset(CIFARDataset):
    """CIFAR-100数据集"""
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform, cifar100=True)


__all__ = [
    'CIFARDataset',
    'CIFAR10Dataset',
    'CIFAR100Dataset',
]

