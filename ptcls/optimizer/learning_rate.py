"""ptcls.optimizer.learning_rate

从 visionhub/ppcls/optimizer/learning_rate.py 的思想移植到 PyTorch：
- 保持“name + 参数”的 YAML 形态
- 支持按 iter 或按 epoch 更新（by_epoch）
- 支持 warmup（线性 warmup）

实现策略：
- 统一返回 torch.optim.lr_scheduler._LRScheduler 或等价对象
- 同时返回一个 lightweight 的包装器，提供 step() 接口，并携带 by_epoch 信息

注意：
PyTorch 的 scheduler 更偏向 step() 被用户调用的位置决定“按 epoch 还是按 iter”。
这里我们通过 wrapper + by_epoch 字段，让 Engine 可以按配置决定何时 step。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _linear_warmup_factor(
    current_step: int,
    warmup_steps: int,
    start_lr: float,
    end_lr: float,
) -> float:
    """返回 warmup 阶段的 lr（绝对值，不是比例）。"""
    if warmup_steps <= 0:
        return end_lr
    if current_step >= warmup_steps:
        return end_lr
    alpha = current_step / float(max(1, warmup_steps))
    return start_lr + alpha * (end_lr - start_lr)


@dataclass
class LRSchedulerWrapper:
    """统一封装，提供 by_epoch 语义并兼容 torch scheduler。"""

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    by_epoch: bool

    def step(self):
        if self.scheduler is None:
            return
        self.scheduler.step()

    def get_last_lr(self):
        if self.scheduler is None:
            return None
        return self.scheduler.get_last_lr()


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """给任意 scheduler 增加线性 warmup。

    - warmup_steps 内使用 start_lr -> base_lr 的线性插值
    - warmup 结束后交给 after_scheduler

    说明：
    - 这里通过修改 optimizer.param_groups['lr'] 实现。
    - after_scheduler 的 base_lrs 使用 warmup 结束时的 base_lr。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_start_lr: float,
        after_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_lr = float(warmup_start_lr)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # last_epoch 在 _LRScheduler 中代表已 step 的次数
        step_idx = self.last_epoch
        base_lrs = self.base_lrs

        if self.warmup_steps > 0 and step_idx < self.warmup_steps:
            return [
                _linear_warmup_factor(step_idx, self.warmup_steps, self.warmup_start_lr, bl)
                for bl in base_lrs
            ]

        # warmup 结束
        if self.after_scheduler is None:
            return base_lrs

        if not self.finished:
            # 让 after_scheduler 从当前 lr 开始
            self.after_scheduler.base_lrs = base_lrs
            self.finished = True

        return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.finished and self.after_scheduler is not None:
            # 交给 after_scheduler
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


# --------- LR builders (name 同 visionhub) ---------

def ConstLR(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_each_epoch: int,
    learning_rate: float,
    warmup_epoch: int = 0,
    warmup_start_lr: float = 0.0,
    last_epoch: int = -1,
    by_epoch: bool = False,
    **kwargs,
) -> LRSchedulerWrapper:
    # 常量 lr：不需要 after_scheduler
    warmup_steps = warmup_epoch if by_epoch else round(warmup_epoch * step_each_epoch)
    sched = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        after_scheduler=None,
        last_epoch=last_epoch,
    )
    return LRSchedulerWrapper(scheduler=sched, by_epoch=by_epoch)


def Linear(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_each_epoch: int,
    learning_rate: float,
    end_lr: float = 0.0,
    power: float = 1.0,
    cycle: bool = False,  # 预留，暂不实现 cycle
    warmup_epoch: int = 0,
    warmup_start_lr: float = 0.0,
    last_epoch: int = -1,
    by_epoch: bool = False,
    **kwargs,
) -> LRSchedulerWrapper:
    # PolynomialDecay 对应：用 LambdaLR 实现
    total_steps = (epochs - warmup_epoch) if by_epoch else (epochs - warmup_epoch) * step_each_epoch
    total_steps = max(1, int(total_steps))

    def poly_lambda(step: int) -> float:
        # step 从 0 开始
        if step >= total_steps:
            return end_lr / learning_rate
        pct = step / float(total_steps)
        lr_val = (learning_rate - end_lr) * ((1 - pct) ** power) + end_lr
        return lr_val / learning_rate

    after = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda, last_epoch=-1)

    warmup_steps = warmup_epoch if by_epoch else round(warmup_epoch * step_each_epoch)
    sched = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        after_scheduler=after,
        last_epoch=last_epoch,
    )
    return LRSchedulerWrapper(scheduler=sched, by_epoch=by_epoch)


def Cosine(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_each_epoch: int,
    learning_rate: float,
    eta_min: float = 0.0,
    warmup_epoch: int = 0,
    warmup_start_lr: float = 0.0,
    last_epoch: int = -1,
    by_epoch: bool = False,
    **kwargs,
) -> LRSchedulerWrapper:
    T_max = (epochs - warmup_epoch) if by_epoch else (epochs - warmup_epoch) * step_each_epoch
    T_max = max(1, int(T_max))

    after = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=-1,
    )

    warmup_steps = warmup_epoch if by_epoch else round(warmup_epoch * step_each_epoch)
    sched = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        after_scheduler=after,
        last_epoch=last_epoch,
    )
    return LRSchedulerWrapper(scheduler=sched, by_epoch=by_epoch)


def Step(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_each_epoch: int,
    learning_rate: float,
    step_size: int = 30,
    gamma: float = 0.1,
    warmup_epoch: int = 0,
    warmup_start_lr: float = 0.0,
    last_epoch: int = -1,
    by_epoch: bool = True,
    **kwargs,
) -> LRSchedulerWrapper:
    # Step 通常按 epoch
    if not by_epoch:
        # 如果用户强行按 iter，就把 step_size(epoch) 转成 iter
        step_size = int(step_size * step_each_epoch)

    after = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=-1,
    )

    warmup_steps = warmup_epoch if by_epoch else round(warmup_epoch * step_each_epoch)
    sched = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        after_scheduler=after,
        last_epoch=last_epoch,
    )
    return LRSchedulerWrapper(scheduler=sched, by_epoch=by_epoch)
