"""ptcls.det

Detection 预测器封装。

目前提供 ultralytics YOLO 版本的 det predictor，用于 shitu system pipeline：
- 输入 RGB ndarray
- 输出 bbox 列表（xyxy）+ score + class_id + class_name

后续如果你需要对齐 visionhub 的 det preprocess/postprocess 细节，可以继续扩展。
"""

from .yolo_predictor import YOLODetPredictor

__all__ = [
    "YOLODetPredictor",
]
