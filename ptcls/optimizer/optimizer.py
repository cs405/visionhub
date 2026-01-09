"""ptcls.optimizer.optimizer

PyTorch 优化器构建器（对应 visionhub/ppcls/optimizer/optimizer.py）。

visionhub 里把每种 optimizer 包装成可调用对象，这里也保持类似风格：
- 接收 learning_rate（float），weight_decay 等超参
- __call__(model_list) 返回 torch.optim.Optimizer

另外补充 visionhub 的一些功能点：
- no_weight_decay_name：根据参数名关键字做 weight decay 分组
- one_dim_param_no_weight_decay：对 1D 参数（常见 bias、LayerNorm weight）禁用 wd（可选）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


def _collect_parameters(model_list: Optional[Sequence[torch.nn.Module]]):
    if model_list is None:
        return None
    params = []
    for m in model_list:
        params.extend(list(m.parameters()))
    return params


def _group_params_for_weight_decay(
    model_list: Sequence[torch.nn.Module],
    weight_decay: float,
    no_weight_decay_name: Optional[str] = None,
    one_dim_param_no_weight_decay: bool = False,
):
    """按名称与维度把参数分成 decay / no_decay 两组。"""
    nd_list = no_weight_decay_name.split() if no_weight_decay_name else []
    params_with_decay = []
    params_without_decay = []

    for m in model_list:
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            hit_name = any(nd in n for nd in nd_list)
            hit_dim = one_dim_param_no_weight_decay and (p.ndim == 1)
            if hit_name or hit_dim:
                params_without_decay.append(p)
            else:
                params_with_decay.append(p)

    if len(params_without_decay) == 0:
        return params_with_decay

    return [
        {"params": params_with_decay, "weight_decay": float(weight_decay)},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]


@dataclass
class SGD:
    learning_rate: float = 0.001
    weight_decay: Optional[float] = None

    def __call__(self, model_list: Optional[Sequence[torch.nn.Module]]):
        params = _collect_parameters(model_list)
        return torch.optim.SGD(
            params,
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay or 0.0),
        )


@dataclass
class Momentum:
    learning_rate: float
    momentum: float
    weight_decay: Optional[float] = None
    use_nesterov: bool = False
    no_weight_decay_name: Optional[str] = None
    one_dim_param_no_weight_decay: bool = False

    def __call__(self, model_list: Optional[Sequence[torch.nn.Module]]):
        if model_list is None:
            raise ValueError("Momentum optimizer in ptcls requires model_list in dynamic graph.")

        wd = float(self.weight_decay or 0.0)
        params = _group_params_for_weight_decay(
            model_list,
            weight_decay=wd,
            no_weight_decay_name=self.no_weight_decay_name,
            one_dim_param_no_weight_decay=self.one_dim_param_no_weight_decay,
        )

        return torch.optim.SGD(
            params,
            lr=float(self.learning_rate),
            momentum=float(self.momentum),
            nesterov=bool(self.use_nesterov),
            weight_decay=wd if isinstance(params, list) else wd,
        )


@dataclass
class Adam:
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: Optional[float] = None

    def __call__(self, model_list: Optional[Sequence[torch.nn.Module]]):
        params = _collect_parameters(model_list)
        return torch.optim.Adam(
            params,
            lr=float(self.learning_rate),
            betas=(float(self.beta1), float(self.beta2)),
            eps=float(self.epsilon),
            weight_decay=float(self.weight_decay or 0.0),
        )


@dataclass
class RMSProp:
    learning_rate: float
    momentum: float = 0.0
    rho: float = 0.95
    epsilon: float = 1e-6
    weight_decay: Optional[float] = None
    no_weight_decay_name: Optional[str] = None
    one_dim_param_no_weight_decay: bool = False

    def __call__(self, model_list: Optional[Sequence[torch.nn.Module]]):
        if model_list is None:
            raise ValueError("RMSProp optimizer in ptcls requires model_list in dynamic graph.")

        wd = float(self.weight_decay or 0.0)
        params = _group_params_for_weight_decay(
            model_list,
            weight_decay=wd,
            no_weight_decay_name=self.no_weight_decay_name,
            one_dim_param_no_weight_decay=self.one_dim_param_no_weight_decay,
        )

        return torch.optim.RMSprop(
            params,
            lr=float(self.learning_rate),
            alpha=float(self.rho),
            eps=float(self.epsilon),
            momentum=float(self.momentum),
            weight_decay=wd if isinstance(params, list) else wd,
        )
