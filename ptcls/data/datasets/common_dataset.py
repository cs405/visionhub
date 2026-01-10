"""Common Dataset Classes

通用数据集类，用于图像分类任务
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CommonDataset(Dataset):
    """通用数据集类

    支持从文件夹结构或列表文件加载数据

    Args:
        root: 数据根目录
        file_list: 文件列表路径（可选）
        transform: 数据变换
        delimiter: 分隔符
    """

    def __init__(self, root, file_list=None, transform=None, delimiter=' '):
        self.root = root
        self.transform = transform
        self.delimiter = delimiter

        self.samples = []
        self.labels = []

        if file_list is not None:
            # 从文件列表加载
            self._load_from_file_list(file_list)
        else:
            # 从文件夹结构加载
            self._load_from_folder()

    def _load_from_file_list(self, file_list):
        """从文件列表加载"""
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(self.delimiter)
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.samples.append(img_path)
                    self.labels.append(label)

    def _load_from_folder(self):
        """从文件夹结构加载"""
        class_dirs = sorted(os.listdir(self.root))

        for label, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root, class_dir)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # 如果路径不是绝对路径，添加root
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImageNetDataset(Dataset):
    """ImageNet数据集类

    Args:
        root: 数据根目录
        split: 'train' 或 'val'
        transform: 数据变换
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(root, split)
        self.samples = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """加载ImageNet数据"""
        class_dirs = sorted(os.listdir(self.data_dir))

        for label, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.JPEG', '.jpg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class MultiLabelDataset(Dataset):
    """多标签数据集类

    Args:
        root: 数据根目录
        file_list: 文件列表路径
        num_classes: 类别数量
        transform: 数据变换
        delimiter: 分隔符
    """

    def __init__(self, root, file_list, num_classes, transform=None, delimiter=' '):
        self.root = root
        self.num_classes = num_classes
        self.transform = transform
        self.delimiter = delimiter

        self.samples = []
        self.labels = []

        self._load_data(file_list)

    def _load_data(self, file_list):
        """加载多标签数据"""
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(self.delimiter)
                img_path = parts[0]

                # 解析多标签
                label_vector = torch.zeros(self.num_classes)
                for label_str in parts[1:]:
                    if label_str:
                        label = int(label_str)
                        if 0 <= label < self.num_classes:
                            label_vector[label] = 1.0

                self.samples.append(img_path)
                self.labels.append(label_vector)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


__all__ = [
    'CommonDataset',
    'ImageNetDataset',
    'MultiLabelDataset',
]

