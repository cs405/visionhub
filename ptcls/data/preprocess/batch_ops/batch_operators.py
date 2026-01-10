# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import random
import numpy as np


class BatchOperator(object):
    """批处理操作基类"""

    def __init__(self, *args, **kwargs):
        pass

    def _unpack(self, batch):
        """解包批次数据"""
        assert isinstance(batch, list), \
            'batch should be a list filled with tuples (img, label)'
        bs = len(batch)
        assert bs > 0, 'size of the batch data should > 0'

        imgs = []
        labels = []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        return np.array(imgs), np.array(labels), bs

    def _one_hot(self, targets):
        """One-hot 编码"""
        return np.eye(self.class_num, dtype="float32")[targets]

    def _mix_target(self, targets0, targets1, lam):
        """混合标签"""
        one_hots0 = self._one_hot(targets0)
        one_hots1 = self._one_hot(targets1)
        return one_hots0 * lam + one_hots1 * (1 - lam)

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """Mixup 数据增强

    Reference: https://arxiv.org/abs/1710.09412

    Args:
        class_num (int): 类别数量
        alpha (float): Beta 分布参数，默认 1.0
    """

    def __init__(self, class_num, alpha: float = 1.):
        super().__init__()
        if alpha <= 0:
            raise ValueError(
                f"Parameter 'alpha' of Mixup should be greater than 0. Got: {alpha}"
            )
        if not class_num:
            raise ValueError("Please set 'class_num' when use MixupOperator")

        self._alpha = alpha
        self.class_num = class_num

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class CutmixOperator(BatchOperator):
    """Cutmix 数据增强

    Reference: https://arxiv.org/abs/1905.04899

    Args:
        class_num (int): 类别数量
        alpha (float): Beta 分布参数，默认 0.2
    """

    def __init__(self, class_num, alpha: float = 0.2):
        super().__init__()
        if alpha <= 0:
            raise ValueError(
                f"Parameter 'alpha' of Cutmix should be greater than 0. Got: {alpha}"
            )
        if not class_num:
            raise ValueError("Please set 'class_num' when use CutmixOperator")

        self._alpha = alpha
        self.class_num = class_num

    def _rand_bbox(self, size, lam):
        """生成随机边界框"""
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # 均匀采样中心点
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]

        # 根据实际裁剪区域调整 lambda
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class FmixOperator(BatchOperator):
    """Fmix 数据增强

    Reference: https://arxiv.org/abs/2002.12047

    Args:
        class_num (int): 类别数量
        alpha (float): 控制混合强度
        decay_power (float): 衰减幂次
        max_soft (float): 最大软化值
        reformulate (bool): 是否重新formulate
    """

    def __init__(
        self,
        class_num,
        alpha: float = 1,
        decay_power: float = 3,
        max_soft: float = 0.,
        reformulate: bool = False
    ):
        super().__init__()
        if not class_num:
            raise ValueError("Please set 'class_num' when use FmixOperator")

        self._alpha = alpha
        self._decay_power = decay_power
        self._max_soft = max_soft
        self._reformulate = reformulate
        self.class_num = class_num

    def _sample_mask(self, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
        """采样 Fmix mask"""
        # 简化实现：生成随机 mask
        # 完整实现需要频域采样，这里使用简化版本
        h, w = shape
        mask = np.random.rand(h, w)
        threshold = np.random.beta(alpha, alpha)
        mask = (mask < threshold).astype(np.float32)

        # 平滑处理
        if max_soft > 0:
            mask = mask * (1 - max_soft) + max_soft / 2

        return threshold, mask.reshape(1, 1, h, w)

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        size = (imgs.shape[2], imgs.shape[3])
        lam, mask = self._sample_mask(
            self._alpha, self._decay_power, size,
            self._max_soft, self._reformulate
        )
        imgs = mask * imgs + (1 - mask) * imgs[idx]
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class HideAndSeek(BatchOperator):
    """Hide-and-Seek 数据增强

    随机隐藏图像的某些区域

    Args:
        class_num (int): 类别数量
        hide_prob (float): 隐藏概率
        grid_size (int): 网格大小
    """

    def __init__(self, class_num, hide_prob: float = 0.5, grid_size: int = 4):
        super().__init__()
        if not class_num:
            raise ValueError("Please set 'class_num' when use HideAndSeek")

        self.class_num = class_num
        self.hide_prob = hide_prob
        self.grid_size = grid_size

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)

        _, _, h, w = imgs.shape
        gh, gw = h // self.grid_size, w // self.grid_size

        for i in range(bs):
            for gi in range(self.grid_size):
                for gj in range(self.grid_size):
                    if np.random.rand() < self.hide_prob:
                        y1, y2 = gi * gh, (gi + 1) * gh
                        x1, x2 = gj * gw, (gj + 1) * gw
                        imgs[i, :, y1:y2, x1:x2] = 0

        return list(zip(imgs, labels))


class GridMask(BatchOperator):
    """GridMask 数据增强

    Reference: https://arxiv.org/abs/2001.04086

    Args:
        class_num (int): 类别数量
        use_h (bool): 是否使用水平网格
        use_w (bool): 是否使用垂直网格
        rotate (int): 旋转角度
        offset (bool): 是否使用偏移
        ratio (float): 网格比率
        mode (int): 模式
        prob (float): 应用概率
    """

    def __init__(
        self,
        class_num,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        prob: float = 0.7
    ):
        super().__init__()
        if not class_num:
            raise ValueError("Please set 'class_num' when use GridMask")

        self.class_num = class_num
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch

        imgs, labels, bs = self._unpack(batch)
        _, _, h, w = imgs.shape

        # 简化实现：生成网格 mask
        d = np.random.randint(2, min(h, w) // 4)
        l = int(d * self.ratio)

        mask = np.ones((h, w), dtype=np.float32)
        for i in range(0, h, d):
            for j in range(0, w, d):
                y1, y2 = i, min(i + l, h)
                x1, x2 = j, min(j + l, w)
                mask[y1:y2, x1:x2] = 0

        mask = mask.reshape(1, 1, h, w)
        imgs = imgs * mask

        return list(zip(imgs, labels))


class OpSampler(object):
    """操作采样器

    从多个操作中随机采样一个应用

    Args:
        class_num (int): 类别数量
        **op_dict: 操作字典，每个操作包含 'prob' 参数
    """

    def __init__(self, class_num, **op_dict):
        if not class_num:
            raise ValueError("Please set 'class_num' when use OpSampler")

        if len(op_dict) < 1:
            print("Warning: No operator in OpSampler. OpSampler has been skipped.")

        self.ops = {}
        total_prob = 0

        for op_name in op_dict:
            param = op_dict[op_name]
            if "prob" not in param:
                print(f"Warning: Parameter 'prob' not set for {op_name}. Using prob=0")

            prob = param.pop("prob", 0)
            total_prob += prob
            param.update({"class_num": class_num})

            # 动态创建操作
            op = globals()[op_name](**param)
            self.ops.update({op: prob})

        if total_prob > 1:
            raise ValueError("The total prob of operators in OpSampler should be <= 1")

        # 添加 None 操作（不做任何处理）
        self.ops[None] = 1 - total_prob

    def __call__(self, batch):
        op = random.choices(
            list(self.ops.keys()),
            weights=list(self.ops.values()),
            k=1
        )[0]
        # None 操作直接返回原始 batch
        return op(batch) if op else batch

