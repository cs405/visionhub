"""ptcls.utils.io

一些简单 I/O 工具：
- 安全创建目录
- 保存 JSON

保持依赖极简，便于部署。
"""

from __future__ import annotations

import json
import os
from typing import Any


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, data: Any, indent: int = 2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
