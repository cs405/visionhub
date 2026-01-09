"""ptcls.rec

Rec(embedding) 推理相关模块。

目标：对齐 visionhub 的 deploy/python/predict_rec.py：
- 支持 transform_ops 预处理
- batch 推理输出 embedding
- 可选 L2 normalize

训练侧（metric learning、人脸、蒸馏等）后续会继续补齐。
"""

from .predictor import RecPredictor

__all__ = [
    "RecPredictor",
]

