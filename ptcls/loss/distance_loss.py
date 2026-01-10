"""Distance Loss Functions

完整迁移自PaddleClas DistanceLoss
"""

import torch.nn as nn


class DistanceLoss(nn.Module):
    """Distance Loss

    支持多种距离损失模式：L1, L2 (MSE), Smooth L1

    Args:
        mode: 损失模式 ('l1', 'l2', 'smooth_l1')
        reduction: 聚合方式 ('mean', 'sum', 'none')
    """

    def __init__(self, mode="l2", reduction='mean', **kwargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"], f"Unsupported mode: {mode}"

        self.mode = mode
        if mode == "l1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(reduction=reduction)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, x, y):
        """
        Args:
            x: (B, D) 预测特征
            y: (B, D) 目标特征

        Returns:
            loss: 标量损失值
        """
        loss = self.loss_func(x, y)
        return loss


__all__ = ['DistanceLoss']

