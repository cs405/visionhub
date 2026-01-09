"""ptcls.optimizer.builder

对齐 visionhub 的 build_optimizer/build_lr_scheduler 入口。

实现约束（最小可用）：
- 支持 config 为 dict：{"name": "Momentum", ...}
- 支持 config 为 list：[{"Momentum": {"scope": "all", ...}}]
- 目前 scope 先只实现 "all"（Engine 暂时也只用单模型）

返回：
- optimizer: torch.optim.Optimizer
- lr_scheduler: ptcls.optimizer.learning_rate.LRSchedulerWrapper 或 None
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Union

import torch

from ..utils import logger
from . import optimizer as optim_mod
from . import learning_rate as lr_mod


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_config: Dict[str, Any],
    epochs: int,
    step_each_epoch: int,
):
    if lr_config is None:
        return None

    lr_cfg = copy.deepcopy(lr_config)

    # visionhub 不写 name 时允许直接给 learning_rate
    if "name" not in lr_cfg:
        return None

    lr_name = lr_cfg.pop("name")
    lr_cfg.update({"epochs": epochs, "step_each_epoch": step_each_epoch})

    if not hasattr(lr_mod, lr_name):
        raise NotImplementedError(f"LR scheduler {lr_name} not implemented in ptcls")

    builder = getattr(lr_mod, lr_name)
    sched = builder(optimizer=optimizer, **lr_cfg)
    return sched


def build_optimizer(
    config: Union[Dict[str, Any], List[Dict[str, Dict[str, Any]]]],
    model: torch.nn.Module,
    epochs: int,
    step_each_epoch: int,
):
    """构建 optimizer + lr scheduler。

    为了先跑通闭环：
    - 仅支持单 optimizer
    - 仅支持 scope=all
    """

    optim_config = copy.deepcopy(config)

    if isinstance(optim_config, dict):
        # {'name': xxx, **optim_cfg} → [{xxx: {'scope': 'all', **optim_cfg}}]
        optim_name = optim_config.pop("name")
        optim_config = [{optim_name: {"scope": "all", **optim_config}}]

    if not isinstance(optim_config, list) or len(optim_config) == 0:
        raise ValueError("Optimizer config must be dict or non-empty list")

    if len(optim_config) != 1:
        logger.warning("ptcls currently supports only 1 optimizer; extra configs will be ignored")

    item = optim_config[0]
    name = list(item.keys())[0]
    cfg = item[name]
    scope = cfg.pop("scope", "all")
    if scope != "all":
        raise NotImplementedError("ptcls optimizer builder currently supports scope=all only")

    lr_cfg = cfg.pop("lr", None)

    if not hasattr(optim_mod, name):
        raise NotImplementedError(f"Optimizer {name} not implemented in ptcls")

    # learning_rate 的来源：
    # - lr_cfg 里可能含 learning_rate
    # - 或者用户直接在 Optimizer 里写 learning_rate
    base_lr = None
    if isinstance(lr_cfg, dict) and "learning_rate" in lr_cfg:
        base_lr = lr_cfg.get("learning_rate")
    else:
        base_lr = cfg.get("learning_rate", cfg.get("lr", None))

    if base_lr is None:
        raise ValueError("Optimizer lr is missing: expected Optimizer.lr.learning_rate or Optimizer.learning_rate")

    cfg = {"learning_rate": float(base_lr), **cfg}

    opt_builder = getattr(optim_mod, name)(**cfg)
    optimizer = opt_builder(model_list=[model])

    lr_scheduler = None
    if isinstance(lr_cfg, dict) and "name" in lr_cfg:
        lr_scheduler = build_lr_scheduler(
            optimizer=optimizer,
            lr_config=lr_cfg,
            epochs=epochs,
            step_each_epoch=step_each_epoch,
        )

        # 如果 scheduler 生效，通常 optimizer 的初始 lr 也应该是 learning_rate
        # WarmupScheduler 会覆盖 early step，无需额外设置。

    return optimizer, lr_scheduler
