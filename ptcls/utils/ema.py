import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class ExponentialMovingAverage(object):
    """指数移动平均（EMA）模型权重平滑

    在训练过程中维护模型参数的移动平均版本。
    EMA 模型通常能提供更稳定的预测结果。

    Args:
        model (nn.Module): 要应用 EMA 的模型
        decay (float): 衰减率，默认 0.9999
        device (str): 设备，默认 None（使用模型当前设备）
        use_num_updates (bool): 是否使用更新次数调整衰减率

    Example:
        >>> model = MyModel()
        >>> ema = ExponentialMovingAverage(model, decay=0.9999)
        >>>
        >>> for epoch in range(num_epochs):
        >>>     for data in dataloader:
        >>>         # 正常训练
        >>>         optimizer.zero_grad()
        >>>         loss = criterion(model(data), target)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>
        >>>         # 更新 EMA
        >>>         ema.update()
        >>>
        >>>     # 使用 EMA 模型进行验证
        >>>     ema.apply_shadow()
        >>>     evaluate(model)
        >>>     ema.restore()
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
        use_num_updates: bool = True
    ):
        self.model = model
        self.decay = decay
        self.device = device
        self.use_num_updates = use_num_updates

        # 保存 EMA 参数的影子副本
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0

        # 初始化影子参数
        self.register()

    def register(self):
        """注册模型参数到影子副本"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新 EMA 参数"""
        if self.use_num_updates:
            self.num_updates += 1
            # 动态调整衰减率
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        # 更新影子参数
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not in shadow"
                    new_average = (
                        decay * self.shadow[name] + (1.0 - decay) * param.data
                    )
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用 EMA 参数到模型（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """返回 EMA 状态字典"""
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        """加载 EMA 状态字典"""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow = state_dict['shadow']

    def to(self, device):
        """移动 EMA 参数到指定设备"""
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self


class ModelEMA(object):
    """简化版 EMA（兼容 YOLO 等模型）

    Args:
        model (nn.Module): 要应用 EMA 的模型
        decay (float): 衰减率
        tau (int): 更新周期
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000):
        # 创建 EMA 模型（深拷贝）
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0

        # 禁用 EMA 模型的梯度计算
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """更新 EMA 模型"""
        self.updates += 1
        # 计算当前衰减率
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.tau)))

        with torch.no_grad():
            # 更新 EMA 参数
            for ema_param, model_param in zip(
                self.ema.parameters(), model.parameters()
            ):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    def update_attr(self, model):
        """更新 EMA 模型的属性"""
        for k, v in model.__dict__.items():
            if not k.startswith('_') and k != 'ema':
                setattr(self.ema, k, v)

