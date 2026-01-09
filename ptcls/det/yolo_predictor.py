"""ptcls.det.yolo_predictor

Ultralytics YOLO 检测封装，用于替代 visionhub 的 DetPredictor。

设计目标：
- 系统 pipeline 只依赖一个稳定的接口：predict(img_rgb) -> List[dict]
- 不在这里做任何与检索相关的逻辑（crop/embedding/search 都在 SystemPredictor）

返回的每个 dict：
- bbox: np.ndarray shape [4], (xmin, ymin, xmax, ymax)
- score: float
- class_id: int
- class_name: str

注意：
- ultralytics 的 model 本身在仓库根目录已有（ultralytics/），无需额外 pip install。
- 这里默认使用 conf 阈值过滤。
"""

from __future__ import annotations

from typing import Any, Dict, List

import os
import sys
import numpy as np

from ..utils import logger


class YOLODetPredictor:
    def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.25, iou: float = 0.7):
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)

        # Try import ultralytics.
        # If user didn't `pip install ultralytics`, fall back to the vendored `ultralytics/` in repo root.
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model_path)
        # ultralytics 支持 .to(device)，但不同版本可能返回 None
        try:
            self.model.to(device)
        except Exception:
            logger.warning("YOLO model .to(device) failed, continue with default device")

    def predict(self, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a single RGB image."""
        if img_rgb is None:
            return []

        # ultralytics 接收 numpy(BGR/RGB 都可以)，但我们统一传 RGB
        results = self.model.predict(img_rgb, conf=self.conf, iou=self.iou, verbose=False)
        if not results:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        out: List[Dict[str, Any]] = []
        for b, s, cid in zip(boxes, scores, cls_ids):
            out.append(
                {
                    "bbox": b.astype(np.float32),
                    "score": float(s),
                    "class_id": int(cid),
                    "class_name": str(r.names.get(int(cid), str(cid))),
                }
            )
        return out
