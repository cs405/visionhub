"""Final Niche Datasets

最后的小众专用数据集
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    """动物识别数据集

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
        self.species = []

        self._load_data()

    def _load_data(self):
        """加载动物数据"""
        data_dir = os.path.join(self.root, self.split)

        if not os.path.exists(data_dir):
            data_dir = self.root

        for label, species_name in enumerate(sorted(os.listdir(data_dir))):
            species_dir = os.path.join(data_dir, species_name)
            if not os.path.isdir(species_dir):
                continue

            for img_name in os.listdir(species_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(species_dir, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)
                    self.species.append(species_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PlantDataset(Dataset):
    """植物识别数据集

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

                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root, img_path)

                self.samples.append(img_path)
                self.labels.append(label)

    def _load_from_folder(self):
        """从文件夹结构加载"""
        for label, plant_name in enumerate(sorted(os.listdir(self.root))):
            plant_dir = os.path.join(self.root, plant_name)
            if not os.path.isdir(plant_dir):
                continue

            for img_name in os.listdir(plant_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(plant_dir, img_name)
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


__all__ = [
    'AnimalDataset',
    'PlantDataset',
]

