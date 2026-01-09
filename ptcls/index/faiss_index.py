"""ptcls.index.faiss_index

Faiss 索引管理器（对齐 visionhub build_gallery.py 中的逻辑，但以可复用类的形式提供）。

约定：
- dist_type: "IP" | "L2" | "hamming"(预留)
- index_method: Faiss index_factory 字符串，例如 "HNSW32" / "IVF1024,Flat" / "Flat"

注意：
- Faiss 的 HNSW binary/remove 限制与 visionhub 相同。
- 当前先完整实现 float 索引；binary/hamming 先把接口留好。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class IndexFiles:
    index_path: str
    id_map_path: str


def _default_files(index_dir: str) -> IndexFiles:
    return IndexFiles(
        index_path=os.path.join(index_dir, "vector.index"),
        id_map_path=os.path.join(index_dir, "id_map.pkl"),
    )


class FaissIndexManager:
    def __init__(self, index_dir: str, dist_type: str = "IP"):
        self.index_dir = index_dir
        self.dist_type = dist_type

    def load(self):
        import faiss

        files = _default_files(self.index_dir)
        if not os.path.exists(files.index_path):
            raise FileNotFoundError(f"vector.index not found: {files.index_path}")
        if not os.path.exists(files.id_map_path):
            raise FileNotFoundError(f"id_map.pkl not found: {files.id_map_path}")

        if self.dist_type == "hamming":
            index = faiss.read_index_binary(files.index_path)
        else:
            index = faiss.read_index(files.index_path)

        with open(files.id_map_path, "rb") as f:
            id_map = pickle.load(f)

        if hasattr(index, "ntotal") and index.ntotal != len(id_map):
            raise ValueError(
                f"Index ntotal ({index.ntotal}) != id_map size ({len(id_map)})"
            )

        return index, id_map

    def save(self, index, id_map: Dict[int, str]):
        import faiss

        os.makedirs(self.index_dir, exist_ok=True)
        files = _default_files(self.index_dir)

        if self.dist_type == "hamming":
            faiss.write_index_binary(index, files.index_path)
        else:
            faiss.write_index(index, files.index_path)

        with open(files.id_map_path, "wb") as f:
            pickle.dump(id_map, f)

    def create(self, embedding_size: int, index_method: str):
        import faiss

        os.makedirs(self.index_dir, exist_ok=True)

        if self.dist_type == "hamming":
            # 预留：binary index
            index_method = "B" + index_method if not index_method.startswith("B") else index_method
            index = faiss.index_binary_factory(embedding_size, index_method)
            id_map: Dict[int, str] = {}
            return index, id_map

        metric = faiss.METRIC_INNER_PRODUCT if self.dist_type == "IP" else faiss.METRIC_L2
        base_index = faiss.index_factory(embedding_size, index_method, metric)

        # 对于多数 float index，我们最终都希望使用顺序 id 维护 id_map。
        # 只有在底层 index 明确支持 IDMap 的情况下才使用 add_with_ids。
        index = base_index
        id_map: Dict[int, str] = {}
        return index, id_map

    def add(
        self,
        index,
        id_map: Dict[int, str],
        features: np.ndarray,
        docs: list[str],
        operation: str = "new",
    ):
        start_id = max(id_map.keys()) + 1 if id_map else 0
        ids_now = (np.arange(0, len(docs)) + start_id).astype(np.int64)

        if self.dist_type == "hamming":
            # binary index: add 不支持 add_with_ids
            index.add(features)
            for i, d in zip(list(range(start_id, start_id + len(docs))), docs):
                id_map[int(i)] = d
            return index, id_map

        if operation == "new" and hasattr(index, "train") and getattr(index, "is_trained", True) is False:
            index.train(features)

        # 关键修复：
        # HNSW32 等索引虽然在 Python 暴露 add_with_ids，但很多情况下并不支持 ID 写入。
        # 我们这里统一采用 index.add() 顺序写入，与 id_map 顺序 ids 对齐。
        index.add(features.astype(np.float32))
        for i, d in zip(list(ids_now), docs):
            id_map[int(i)] = d

        return index, id_map

    def remove(self, index, id_map: Dict[int, str], docs_to_remove: list[str]):
        if self.dist_type == "hamming":
            raise RuntimeError("Binary/Hamming index remove is not implemented")

        remove_ids = [k for k, v in id_map.items() if v in docs_to_remove]
        if not remove_ids:
            return index, id_map

        import numpy as np

        index.remove_ids(np.asarray(remove_ids, dtype=np.int64))
        for k in remove_ids:
            id_map.pop(int(k), None)
        return index, id_map

    def search(self, index, query_features: np.ndarray, topk: int):
        # query_features shape: [B, D]
        if self.dist_type == "hamming":
            return index.search(query_features, topk)
        return index.search(query_features.astype(np.float32), topk)
