# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union


class Topk(object):
    """Top-K 分类结果后处理

    Args:
        topk (int): 返回 Top-K 个结果，默认为 1
        class_id_map_file (str): 类别名映射文件路径
        delimiter (str): 分隔符，默认为空格
        label_list (list): 类别名列表，优先于 class_id_map_file
    """

    def __init__(
        self,
        topk: int = 1,
        class_id_map_file: Optional[str] = None,
        delimiter: Optional[str] = None,
        label_list: Optional[List[str]] = None
    ):
        assert isinstance(topk, int), "topk must be int"
        self.topk = topk
        self.delimiter = delimiter if delimiter is not None else " "

        # label_list 优先于 class_id_map_file
        if label_list is not None:
            self.class_id_map = label_list
        else:
            self.class_id_map = self.parse_class_id_map(class_id_map_file)

    def parse_class_id_map(self, class_id_map_file):
        """解析类别名映射文件"""
        if class_id_map_file is None:
            return None

        if not os.path.exists(class_id_map_file):
            print(
                "Warning: If want to use your own label_dict, please input legal path!\n"
                "Otherwise label_names will be empty!"
            )
            return None

        try:
            class_id_map = {}
            # 尝试 UTF-8 编码
            try:
                with open(class_id_map_file, 'r', encoding='utf-8') as fin:
                    lines = fin.readlines()
            except Exception:
                # 回退到 GBK 编码
                with open(class_id_map_file, 'r', encoding='gbk') as fin:
                    lines = fin.readlines()

            for line in lines:
                partition = line.split("\n")[0].partition(self.delimiter)
                class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            print(f"Error parsing class_id_map_file: {ex}")
            class_id_map = None

        return class_id_map

    def __call__(
        self,
        x: Union[torch.Tensor, np.ndarray, dict],
        file_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """执行后处理

        Args:
            x: 模型输出 logits，shape [B, num_classes]
               可以是 torch.Tensor, np.ndarray 或包含 'logits' 的字典
            file_names: 文件名列表

        Returns:
            结果列表，每个元素为字典:
            {
                'class_ids': [id1, id2, ...],
                'scores': [score1, score2, ...],
                'label_names': [name1, name2, ...],  # 可选
                'file_name': 'xxx.jpg'  # 可选
            }
        """
        # 处理字典输入
        if isinstance(x, dict):
            x = x['logits']

        # 转换为 numpy array
        if isinstance(x, torch.Tensor):
            x = F.softmax(x, dim=-1)
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            # 如果已经是 numpy array，假设未经过 softmax
            if x.max() > 1.0 or x.min() < 0.0:
                # 需要 softmax
                x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
                x = x_exp / np.sum(x_exp, axis=-1, keepdims=True)

        if file_names is not None:
            assert x.shape[0] == len(file_names), \
                f"Batch size {x.shape[0]} != file_names length {len(file_names)}"

        y = []
        for idx, probs in enumerate(x):
            # 获取 Top-K 索引
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")

            clas_id_list = []
            score_list = []
            label_name_list = []

            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    if isinstance(self.class_id_map, dict):
                        label_name_list.append(self.class_id_map.get(i.item(), f"class_{i}"))
                    elif isinstance(self.class_id_map, list):
                        label_name_list.append(
                            self.class_id_map[i.item()] if i.item() < len(self.class_id_map)
                            else f"class_{i}"
                        )

            result = {
                "class_ids": clas_id_list,
                "scores": np.around(score_list, decimals=5).tolist(),
            }

            if file_names is not None:
                result["file_name"] = file_names[idx]

            if label_name_list:
                result["label_names"] = label_name_list

            y.append(result)

        return y

