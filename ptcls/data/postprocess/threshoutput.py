# Migrated to PyTorch for visionhub project.

import os
import numpy as np
from typing import List, Dict, Optional


def parse_class_id_map(class_id_map_file, delimiter):
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
        with open(class_id_map_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                partition = line.split("\n")[0].partition(delimiter)
                class_id_map[int(partition[0])] = str(partition[-1])
    except Exception as ex:
        print(ex)
        class_id_map = None
    return class_id_map


class ThreshOutput(object):
    """阈值输出后处理（二分类/多分类）

    Args:
        threshold (float): 阈值，默认 0
        default_label_index (int): 默认标签索引，默认 0
        label_0 (str): 标签 0 的名称
        label_1 (str): 标签 1 的名称
        class_id_map_file (str): 类别名映射文件路径
        delimiter (str): 分隔符
    """

    def __init__(
        self,
        threshold: float = 0,
        default_label_index: int = 0,
        label_0: str = "0",
        label_1: str = "1",
        class_id_map_file: Optional[str] = None,
        delimiter: Optional[str] = None
    ):
        self.threshold = threshold
        self.default_label_index = default_label_index
        self.label_0 = label_0
        self.label_1 = label_1

        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(class_id_map_file, delimiter)

    def __call__(self, x, file_names=None):
        """执行后处理

        Args:
            x: 模型输出概率，shape [B, num_classes]
            file_names: 文件名列表

        Returns:
            结果列表
        """
        def binary_classification(x):
            """二分类处理"""
            y = []
            for idx, probs in enumerate(x):
                score = probs[1]
                if score < self.threshold:
                    result = {
                        "class_ids": [0],
                        "scores": [1 - score],
                        "label_names": [self.label_0]
                    }
                else:
                    result = {
                        "class_ids": [1],
                        "scores": [score],
                        "label_names": [self.label_1]
                    }
                if file_names is not None:
                    result["file_name"] = file_names[idx]
                y.append(result)
            return y

        def multi_classification(x):
            """多分类处理"""
            y = []
            for idx, probs in enumerate(x):
                index = probs.argsort(axis=0)[::-1].astype("int32")
                top1_id = index[0]
                top1_score = probs[top1_id]

                if top1_score > self.threshold:
                    rtn_id = top1_id
                else:
                    rtn_id = self.default_label_index

                label_name = (
                    self.class_id_map[rtn_id]
                    if self.class_id_map is not None
                    else ""
                )

                result = {
                    "class_ids": [rtn_id],
                    "scores": [probs[rtn_id]],
                    "label_names": [label_name]
                }
                if file_names is not None:
                    result["file_name"] = file_names[idx]
                y.append(result)
            return y

        if file_names is not None:
            assert x.shape[0] == len(file_names)

        if x.shape[1] == 2:
            return binary_classification(x)
        else:
            return multi_classification(x)


class MultiLabelThreshOutput(object):
    """多标签阈值输出后处理

    Args:
        threshold (float): 阈值，默认 0.5
        class_id_map_file (str): 类别名映射文件路径
        delimiter (str): 分隔符
    """

    def __init__(
        self,
        threshold: float = 0.5,
        class_id_map_file: Optional[str] = None,
        delimiter: Optional[str] = None
    ):
        self.threshold = threshold
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(class_id_map_file, delimiter)

    def __call__(self, x, file_names=None):
        """执行后处理

        Args:
            x: 模型输出概率，shape [B, num_classes]
            file_names: 文件名列表

        Returns:
            结果列表
        """
        y = []
        for idx, probs in enumerate(x):
            index = np.where(probs >= self.threshold)[0].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []

            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])

            result = {
                "class_ids": clas_id_list,
                "scores": np.around(score_list, decimals=5).tolist(),
                "label_names": label_name_list
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y

