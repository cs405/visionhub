import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..preprocess import create_operators, transform
from ...utils import logger

class CommonDataset(Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self._transform_ops = create_operators(transform_ops).ops if transform_ops else None

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):
        pass

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            # 默认使用 DecodeImage 所以这里直接传路径
            img = image_path
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            
            label = self.labels[idx]
            return img, label

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))

class ImageNetDataset(CommonDataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=None,
                 relabel=False):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        super(ImageNetDataset, self).__init__(image_root, cls_label_path,
                                              transform_ops)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if self.relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split(self.delimiter)
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(sorted(label_set))
                }

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
