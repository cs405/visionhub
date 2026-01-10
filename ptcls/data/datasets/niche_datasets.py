"""Specialized Niche Datasets

特定领域的小众数据集
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class ProductDataset(Dataset):
    """商品识别数据集

    Args:
        root: 数据根目录
        file_list: 文件列表路径
        transform: 数据变换
    """

    def __init__(self, root, file_list=None, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []
        self.labels = []
        self.product_ids = []

        if file_list is not None:
            self._load_from_file(file_list)
        else:
            self._load_from_folder()

    def _load_from_file(self, file_list):
        """从文件列表加载"""
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                img_path = parts[0]
                label = int(parts[1])
                product_id = parts[2] if len(parts) > 2 else str(label)

                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root, img_path)

                self.samples.append(img_path)
                self.labels.append(label)
                self.product_ids.append(product_id)

    def _load_from_folder(self):
        """从文件夹结构加载"""
        for label, product_name in enumerate(sorted(os.listdir(self.root))):
            product_dir = os.path.join(self.root, product_name)
            if not os.path.isdir(product_dir):
                continue

            for img_name in os.listdir(product_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(product_dir, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)
                    self.product_ids.append(product_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class FoodDataset(Dataset):
    """食品识别数据集

    Args:
        root: 数据根目录
        split: 'train' 或 'test'
        transform: 数据变换
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """加载食品数据"""
        data_dir = os.path.join(self.root, self.split)

        # 尝试读取标注文件
        label_file = os.path.join(self.root, f'{self.split}_labels.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    img_path = os.path.join(data_dir, parts[0])
                    label = int(parts[1])

                    self.samples.append(img_path)
                    self.labels.append(label)
        else:
            # 从文件夹结构加载
            for label, food_class in enumerate(sorted(os.listdir(data_dir))):
                class_dir = os.path.join(data_dir, food_class)
                if not os.path.isdir(class_dir):
                    continue

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
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


class LandmarkDataset(Dataset):
    """地标识别数据集

    Args:
        root: 数据根目录
        file_list: 文件列表路径（可选）
        transform: 数据变换
    """

    def __init__(self, root, file_list=None, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []
        self.labels = []
        self.landmark_names = []

        if file_list is not None:
            self._load_from_file(file_list)
        else:
            self._load_from_folder()

    def _load_from_file(self, file_list):
        """从文件列表加载"""
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                img_path = parts[0]
                label = int(parts[1])
                landmark_name = parts[2] if len(parts) > 2 else f"landmark_{label}"

                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root, img_path)

                self.samples.append(img_path)
                self.labels.append(label)
                self.landmark_names.append(landmark_name)

    def _load_from_folder(self):
        """从文件夹结构加载"""
        for label, landmark_name in enumerate(sorted(os.listdir(self.root))):
            landmark_dir = os.path.join(self.root, landmark_name)
            if not os.path.isdir(landmark_dir):
                continue

            for img_name in os.listdir(landmark_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(landmark_dir, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)
                    self.landmark_names.append(landmark_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


__all__ = [
    'ProductDataset',
    'FoodDataset',
    'LandmarkDataset',
]

