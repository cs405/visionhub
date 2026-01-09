"""ptcls.data.dataset.yolo_det_crop_dataset

把 YOLO Detection 格式（images/ + labels/*.txt）转换为“用于 embedding 训练”的数据集：按 bbox 裁剪目标。

YOLO label txt 格式：
- 每行：class_id cx cy w h  (均为相对于原图宽高归一化的 float)

本数据集返回：
- crop PIL.Image(RGB)
- class_id (int)

注意：
- 这是检索/识别训练用，不是检测训练用。
- 如果你需要目标检测训练，请直接用 ultralytics 的 yolo train（它会用同样的格式）。

后续可扩展：
- 支持 segmentation label
- 支持 ignore flags
- 支持按类别采样/PKSampler
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ...utils.yolo_data import load_yolo_data_yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class YoloDetSample:
    img_path: str
    image_id: str
    cls_id: int
    cls_name: str
    xyxy: Tuple[int, int, int, int]


class YoloDetCropDataset:
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        data_yaml: Optional[str] = None,
        return_name: bool = False,
    ):
        images_dir = os.path.abspath(images_dir)
        labels_dir = os.path.abspath(labels_dir)

        # Normalize common YOLO layouts.
        # Supported:
        # 1) Standard: images_dir=.../images/train, labels_dir=.../labels/train
        # 2) Split root: images_dir=.../train, labels_dir=.../train  (contains images/ and labels/)
        # 3) Mixed: images_dir=.../images/train, labels_dir=.../train (or vice versa)

        def _try_expand_split_root(img_d: str, lab_d: str):
            cand_img = os.path.join(img_d, "images")
            cand_lab = os.path.join(lab_d, "labels")
            if os.path.isdir(cand_img) and os.path.isdir(cand_lab):
                return cand_img, cand_lab
            return img_d, lab_d

        images_dir, labels_dir = _try_expand_split_root(images_dir, labels_dir)

        # If one side points to a split root, expand both from that split root.
        if not os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            cand_img = os.path.join(labels_dir, "images")
            if os.path.isdir(cand_img):
                images_dir = cand_img
        if not os.path.isdir(labels_dir) and os.path.isdir(images_dir):
            cand_lab = os.path.join(images_dir, "labels")
            if os.path.isdir(cand_lab):
                labels_dir = cand_lab

        # If user passed dataset/images (no train/val), try to infer split from labels_dir.
        # e.g. images_dir=dataset/images/train doesn't exist, but dataset/train/images does.
        if not os.path.isdir(images_dir):
            # Try: <parent>/train/images when given .../images/train that doesn't exist.
            parent = os.path.dirname(os.path.dirname(images_dir))  # .../dataset
            base = os.path.basename(images_dir)
            if base in ("train", "val", "test"):
                # attempt: dataset/<split>/images
                cand = os.path.join(parent, base, "images")
                if os.path.isdir(cand):
                    images_dir = cand

        if not os.path.isdir(labels_dir):
            parent = os.path.dirname(os.path.dirname(labels_dir))
            base = os.path.basename(labels_dir)
            if base in ("train", "val", "test"):
                cand = os.path.join(parent, base, "labels")
                if os.path.isdir(cand):
                    labels_dir = cand

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.return_name = bool(return_name)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(self.images_dir)
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(self.labels_dir)

        self.id2name: Dict[int, str] = {}
        if data_yaml:
            self.id2name, _ = load_yolo_data_yaml(data_yaml)

        self.samples: List[YoloDetSample] = []
        for fn in os.listdir(self.images_dir):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(self.images_dir, fn)
            stem = os.path.splitext(fn)[0]
            lab_path = os.path.join(self.labels_dir, stem + ".txt")
            if not os.path.exists(lab_path):
                continue

            # read image size lazily once
            img = Image.open(img_path)
            w, h = img.size
            img.close()

            with open(lab_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cls_name = self.id2name.get(cls_id, str(cls_id))
                    cx, cy, bw, bh = map(float, parts[1:5])

                    x1 = int((cx - bw / 2.0) * w)
                    y1 = int((cy - bh / 2.0) * h)
                    x2 = int((cx + bw / 2.0) * w)
                    y2 = int((cy + bh / 2.0) * h)

                    # clip
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(1, min(x2, w))
                    y2 = max(1, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    self.samples.append(
                        YoloDetSample(img_path=img_path, image_id=stem, cls_id=cls_id, cls_name=cls_name, xyxy=(x1, y1, x2, y2))
                    )

        if not self.samples:
            raise RuntimeError(f"No yolo det labels found. images_dir={self.images_dir}, labels_dir={self.labels_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        x1, y1, x2, y2 = s.xyxy
        crop = img.crop((x1, y1, x2, y2))
        if self.return_name:
            return crop, s.cls_id, s.cls_name, s.image_id, s.xyxy
        return crop, s.cls_id, s.image_id, s.xyxy
