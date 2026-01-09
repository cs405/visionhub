"""ptcls.optimizer

Professional本的优化器与学习率构建模块。

目标：尽量对齐 visionhub 的 YAML 写法：

Optimizer:
  name: Momentum
  lr:
    name: Cosine
    learning_rate: 0.1
    warmup_epoch: 5
    warmup_start_lr: 0.0
    by_epoch: false
  momentum: 0.9
  weight_decay: 0.0001

当前实现先覆盖最常用的分类训练闭环：
- 优化器：SGD / Momentum / Adam / RMSprop
- 学习率：ConstLR / Linear / Cosine / Step
- Warmup：线性 warmup（可选）

后续如需对齐 visionhub 的更多 lr 策略（Cyclic/OneCycle/ReduceOnPlateau 等），可以继续扩展。
"""

from .builder import build_lr_scheduler, build_optimizer

__all__ = [
    "build_lr_scheduler",
    "build_optimizer",
]

