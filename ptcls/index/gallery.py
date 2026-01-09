"""ptcls.index.gallery

对齐 visionhub/deploy/python/build_gallery.py：
- 读取 data_file（图片路径 + 其它描述信息）
- 批量抽取 embedding
- new/append/remove 操作索引

这里不依赖 visionhubPredictor，而是依赖 ptcls.rec.predictor.RecPredictor（Professional）。
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import logger
from .faiss_index import FaissIndexManager


def split_datafile(data_file: str, image_root: str, delimiter: str = "\t") -> Tuple[List[str], List[str]]:
    """读取 gallery 的 data_file。

    如果 data_file 不存在或为空：返回空列表，让上层走 auto-scan。
    """
    gallery_images: List[str] = []
    gallery_docs: List[str] = []

    if not os.path.exists(data_file):
        return gallery_images, gallery_docs

    with open(data_file, "r", encoding="utf-8") as f:
        for ori_line in f:
            s = ori_line.strip()
            if not s:
                continue
            line = s.split(delimiter)
            if len(line) < 2:
                raise ValueError(f"line({ori_line}) must split into >=2 parts")
            image_file = os.path.join(image_root, line[0])
            gallery_images.append(image_file)
            gallery_docs.append(s)

    return gallery_images, gallery_docs


def _auto_gallery_from_image_root(image_root: str) -> Tuple[List[str], List[str]]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    gallery_images: List[str] = []
    gallery_docs: List[str] = []

    for root, _, files in os.walk(image_root):
        for fn in files:
            if not fn.lower().endswith(exts):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, image_root)
            # label: parent dir name if in subfolder else stem
            parent = os.path.basename(os.path.dirname(full))
            label = parent if parent and parent != os.path.basename(image_root) else os.path.splitext(fn)[0]
            gallery_images.append(full)
            gallery_docs.append(f"{rel}\t{label}")

    return gallery_images, gallery_docs


class GalleryBuilder:
    def __init__(self, config: dict, rec_predictor, config_path: str = None):
        self.config = config
        self.rec_predictor = rec_predictor
        self.config_path = config_path
        if "IndexProcess" not in config:
            raise KeyError("IndexProcess not found in config")

    def _resolve_path(self, p: str) -> str:
        if p is None:
            return p
        if os.path.isabs(p):
            return p

        # 优先使用 Global.workspace_dir（如果配置中提供）
        ws = None
        try:
            ws = self.config.get("Global", {}).get("workspace_dir")
        except Exception:
            ws = None

        if ws:
            base = os.path.abspath(ws)
        else:
            # 默认：仓库根目录 = visionhub 的上一级（与 label_gallery/label_images 同级）
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

        return os.path.abspath(os.path.join(base, p))

    def build(self):
        cfg = self.config["IndexProcess"]
        operation = cfg.get("index_operation", "new").lower()
        if operation not in ["new", "append", "remove"]:
            raise ValueError("Only new/append/remove are supported")

        image_root = self._resolve_path(cfg["image_root"])
        data_file = self._resolve_path(cfg["data_file"])
        index_dir = self._resolve_path(cfg["index_dir"])
        delimiter = cfg.get("delimiter", "\t")

        logger.info(f"[GalleryBuilder] image_root={image_root}")
        logger.info(f"[GalleryBuilder] data_file={data_file}")
        logger.info(f"[GalleryBuilder] index_dir={index_dir}")

        gallery_images, gallery_docs = split_datafile(data_file, image_root, delimiter)
        if len(gallery_images) == 0:
            logger.warning(
                f"[GalleryBuilder] data_file is empty: {data_file}. Fallback to scan image_root={image_root}"
            )
            gallery_images, gallery_docs = _auto_gallery_from_image_root(image_root)

            # persist docs back to data_file
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            with open(data_file, "w", encoding="utf-8") as f:
                for d in gallery_docs:
                    f.write(d + "\n")
            logger.info(f"[GalleryBuilder] persisted auto gallery docs to: {data_file}")

        logger.info(f"[GalleryBuilder] gallery_size={len(gallery_images)}")

        dist_type = cfg.get("dist_type", "IP")
        index_method = cfg.get("index_method", "HNSW32")
        embedding_size = int(cfg["embedding_size"])

        # IVF 自动估计（参考 visionhub）
        if index_method == "IVF":
            nlist = min(max(len(gallery_images) // 8, 1), 65536)
            index_method = f"IVF{nlist},Flat"

        manager = FaissIndexManager(index_dir=index_dir, dist_type=dist_type)

        # remove 不需要提特征
        if operation != "remove":
            feats = self._extract_features(gallery_images, cfg)
            logger.info(
                f"[GalleryBuilder] feats shape={getattr(feats, 'shape', None)}, min={float(feats.min()) if feats.size else 'na'}, max={float(feats.max()) if feats.size else 'na'}"
            )
        else:
            feats = None

        if operation in ["append", "remove"]:
            index, id_map = manager.load()
        else:
            index, id_map = manager.create(embedding_size=embedding_size, index_method=index_method)

        if operation != "remove":
            index, id_map = manager.add(index, id_map, feats, gallery_docs, operation=operation)
        else:
            index, id_map = manager.remove(index, id_map, gallery_docs)

        manager.save(index, id_map)
        logger.info(f"Gallery index saved to: {index_dir}, ntotal={getattr(index, 'ntotal', None)}")

    def _extract_features(self, image_paths: List[str], cfg: dict) -> np.ndarray:
        embedding_size = int(cfg["embedding_size"])
        dist_type = cfg.get("dist_type", "IP")

        if dist_type == "hamming":
            # 预留：binary
            features = np.zeros([len(image_paths), embedding_size // 8], dtype=np.uint8)
        else:
            features = np.zeros([len(image_paths), embedding_size], dtype=np.float32)

        batch_size = int(cfg.get("batch_size", 32))
        batch_imgs = []
        batch_indices = []

        for i, image_file in enumerate(tqdm(image_paths, desc="Extract gallery features")):
            img = cv2.imread(image_file)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image_file}")
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            batch_indices.append(i)

            if len(batch_imgs) >= batch_size:
                rec_feat = self.rec_predictor.predict(batch_imgs)
                features[batch_indices, :] = rec_feat
                batch_imgs, batch_indices = [], []

        if batch_imgs:
            rec_feat = self.rec_predictor.predict(batch_imgs)
            features[batch_indices, :] = rec_feat

        return features
