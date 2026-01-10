# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import os
import shutil
from functools import partial


class SavePreLabel(object):
    """保存预标签（将图片复制到对应类别文件夹）

    Args:
        save_dir (str): 保存目录
    """

    def __init__(self, save_dir: str):
        if save_dir is None:
            raise Exception("Please specify save_dir if SavePreLabel specified.")
        self.save_dir = partial(os.path.join, save_dir)

    def __call__(self, x, file_names=None):
        """执行保存

        Args:
            x: 模型输出概率，shape [B, num_classes]
            file_names: 文件名列表
        """
        if file_names is None:
            return
        assert x.shape[0] == len(file_names)

        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-1].astype("int32")
            self.save(index, file_names[idx])

    def save(self, id, image_file):
        """保存单张图片到对应类别文件夹"""
        output_dir = self.save_dir(str(id))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(image_file, output_dir)

