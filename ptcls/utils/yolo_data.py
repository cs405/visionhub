"""ptcls.utils.yolo_data

读取 Ultralytics/YOLO 的 data.yaml，提供 class_id <-> class_name 映射。

支持两种常见写法：
- names: ["a", "b", ...]
- names:
    0: person
    1: car

输出：
- id2name: dict[int,str]
- name2id: dict[str,int]
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import yaml


def load_yolo_data_yaml(data_yaml: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    data_yaml = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(data_yaml)

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise KeyError(f"names not found in {data_yaml}")

    id2name: Dict[int, str] = {}
    if isinstance(names, list):
        id2name = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        # keys might be int or str
        for k, v in names.items():
            id2name[int(k)] = str(v)
    else:
        raise TypeError(f"Unsupported names type in {data_yaml}: {type(names)}")

    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id

