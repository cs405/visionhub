"""ptcls.index

向量检索索引管理（Professional，对齐 visionhub deploy/python/build_gallery.py 的能力）。

- 支持 Faiss float index（L2/IP）
- 支持 binary/hamming（后续扩展）
- 支持 new/append/remove

当前仓库已经有 label_gallery/index 结构（metadata.pkl + vector.index），这里统一规范：
- vector.index : faiss index 文件
- id_map.pkl   : id -> doc(string) 映射（与 visionhub 一致）
"""

from .gallery import GalleryBuilder
from .faiss_index import FaissIndexManager

__all__ = [
    "GalleryBuilder",
    "FaissIndexManager",
]

