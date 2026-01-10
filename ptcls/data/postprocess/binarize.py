# Migrated to PyTorch for visionhub project.

import numpy as np


class Binarize(object):
    """二值化后处理（用于哈希检索）

    将浮点特征向量转换为二值哈希码

    Args:
        method (str): 二值化方法，'round' 或 'sign'
    """

    def __init__(self, method: str = "round"):
        self.method = method
        self.unit = np.array([[128, 64, 32, 16, 8, 4, 2, 1]]).T

    def __call__(self, x, file_names=None):
        """执行二值化

        Args:
            x: 特征向量，shape [B, D]，D 必须是 8 的倍数
            file_names: 文件名列表（未使用）

        Returns:
            二值化后的字节数组，shape [B, D//8]
        """
        if self.method == "round":
            x = np.round(x + 1).astype("uint8") - 1

        if self.method == "sign":
            x = ((np.sign(x) + 1) / 2).astype("uint8")

        embedding_size = x.shape[1]
        assert (
            embedding_size % 8 == 0
        ), "The Binary index only support vectors with sizes multiple of 8"

        byte = np.zeros([x.shape[0], embedding_size // 8], dtype=np.uint8)
        for i in range(embedding_size // 8):
            byte[:, i:i + 1] = np.dot(x[:, i * 8:(i + 1) * 8], self.unit)

        return byte

