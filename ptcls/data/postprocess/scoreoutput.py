# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import numpy as np
from typing import List, Dict, Optional


class ScoreOutput(object):
    """分数输出后处理

    Args:
        decimal_places (int): 保留的小数位数
    """

    def __init__(self, decimal_places: int = 5):
        self.decimal_places = decimal_places

    def __call__(self, x, file_names=None):
        """执行后处理

        Args:
            x: 模型输出，shape [B, ...]
            file_names: 文件名列表

        Returns:
            结果列表
        """
        y = []
        for idx, probs in enumerate(x):
            score = np.around(x[idx], self.decimal_places)
            result = {"scores": score}
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y

