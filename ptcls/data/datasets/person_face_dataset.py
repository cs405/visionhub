"""Person and Face Datasets

行人重识别和人脸识别专用数据集
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class PersonDataset(Dataset):
    """行人重识别数据集

    适用于Market-1501, DukeMTMC等数据集格式

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
        self.pids = []  # person IDs
        self.camids = []  # camera IDs

        self._load_data()

    def _load_data(self):
        """加载行人数据"""
        data_dir = os.path.join(self.root, self.split)

        if not os.path.exists(data_dir):
            data_dir = self.root

        # 从文件名解析信息
        for img_name in os.listdir(data_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(data_dir, img_name)

            # 解析文件名格式：pid_camid_frameid.jpg
            try:
                parts = img_name.split('_')
                if len(parts) >= 2:
                    pid = int(parts[0])
                    camid = int(parts[1][1:]) if parts[1].startswith('c') else int(parts[1])

                    self.samples.append(img_path)
                    self.pids.append(pid)
                    self.camids.append(camid)
            except:
                continue

    @property
    def labels(self):
        """兼容性属性"""
        return self.pids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        pid = self.pids[idx]
        camid = self.camids[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid


class FaceDataset(Dataset):
    """人脸识别数据集

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
        for label, person_id in enumerate(sorted(os.listdir(self.root))):
            person_dir = os.path.join(self.root, person_id)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
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


class CustomLabelDataset(Dataset):
    """自定义标签数据集

    通用的自定义标签数据集，支持灵活的标签格式

    Args:
        root: 数据根目录
        file_list: 文件列表路径
        transform: 数据变换
        delimiter: 分隔符
        label_col: 标签列索引
    """

    def __init__(self, root, file_list, transform=None, delimiter=' ', label_col=1):
        self.root = root
        self.transform = transform
        self.delimiter = delimiter
        self.label_col = label_col

        self.samples = []
        self.labels = []

        self._load_data(file_list)

    def _load_data(self, file_list):
        """加载数据"""
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(self.delimiter)
                if len(parts) <= self.label_col:
                    continue

                img_path = parts[0]
                label = int(parts[self.label_col])

                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root, img_path)

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
    'PersonDataset',
    'FaceDataset',
    'CustomLabelDataset',
]

