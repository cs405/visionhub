"""Vehicle and Logo Datasets

车辆和Logo专用数据集类
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class CompCarsDataset(Dataset):
    """CompCars车辆数据集

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
        """加载CompCars数据"""
        data_dir = os.path.join(self.root, self.split)

        # 读取标注文件
        label_file = os.path.join(self.root, f'{self.split}_label.txt')

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
            for make_id in os.listdir(data_dir):
                make_dir = os.path.join(data_dir, make_id)
                if not os.path.isdir(make_dir):
                    continue

                for model_id in os.listdir(make_dir):
                    model_dir = os.path.join(make_dir, model_id)
                    if not os.path.isdir(model_dir):
                        continue

                    label = int(make_id) * 1000 + int(model_id)

                    for img_name in os.listdir(model_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(model_dir, img_name)
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


class VeriWildDataset(Dataset):
    """VeriWild车辆重识别数据集

    Args:
        root: 数据根目录
        split: 'train', 'query' 或 'gallery'
        transform: 数据变换
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []
        self.labels = []
        self.camera_ids = []

        self._load_data()

    def _load_data(self):
        """加载VeriWild数据"""
        list_file = os.path.join(self.root, f'{self.split}_list.txt')

        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                img_path = os.path.join(self.root, 'images', parts[0])
                vehicle_id = int(parts[1])
                camera_id = int(parts[2]) if len(parts) > 2 else 0

                self.samples.append(img_path)
                self.labels.append(vehicle_id)
                self.camera_ids.append(camera_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        camera_id = self.camera_ids[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, camera_id


class LogoDataset(Dataset):
    """Logo数据集

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
        for label, logo_name in enumerate(sorted(os.listdir(self.root))):
            logo_dir = os.path.join(self.root, logo_name)
            if not os.path.isdir(logo_dir):
                continue

            for img_name in os.listdir(logo_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(logo_dir, img_name)
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
    'CompCarsDataset',
    'VeriWildDataset',
    'LogoDataset',
]

