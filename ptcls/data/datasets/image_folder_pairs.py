"""ptcls.data.dataset.image_folder_pairs

一个最小的“检索训练数据集”实现：

目录结构示例：
root/
  class_a/
    a1.jpg
    a2.jpg
  class_b/
    b1.jpg

返回：
- image (np RGB)
- label (int)

说明：
- 先对齐 visionhub Shitu 场景的最小训练闭环（分类/对比学习均可）。
- 后续可以扩展为：读取 train_list.txt、支持多标签、支持 hard negative、支持 PKSampler 等。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Sample:
    path: str
    label: int


class ImageFolderDataset:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(self.root_dir)

        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: List[Sample] = []
        for c in classes:
            cdir = os.path.join(self.root_dir, c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append(Sample(path=os.path.join(cdir, fn), label=self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No images found in: {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        return img, s.label

