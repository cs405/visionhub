"""ptcls.utils.visualize

用于 shitu/system predictor 的可视化工具。

输入：
- img_rgb: np.ndarray, shape [H, W, 3], RGB
- results: SystemPredictor.predict() 输出的 list[dict]

输出：
- vis_bgr: np.ndarray (BGR)，可直接 cv2.imwrite

绘制内容：
- bbox
- det_class
- rec_docs
- rec_scores

绘制风格：
- 有匹配：绿色
- 无匹配：红色
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_shitu_results(
    img_rgb: np.ndarray,
    results: List[Dict[str, Any]],
    score_thres: float = 0.0,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    if img_rgb is None:
        raise ValueError("img_rgb is None")

    vis = img_rgb[:, :, ::-1].copy()  # RGB->BGR

    for r in results or []:
        bbox = r.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]

        det_class = r.get("det_class") or "unknown"
        rec_doc = r.get("rec_docs")
        score = float(r.get("rec_scores", 0.0))

        matched = rec_doc is not None and score >= score_thres
        color = (0, 255, 0) if matched else (0, 0, 255)

        # label text
        if rec_doc is None:
            text = f"{det_class} | no_match | {score:.2f}"
        else:
            # 控制文本长度，避免太长覆盖画面
            doc_show = str(rec_doc)
            if len(doc_show) > 40:
                doc_show = doc_show[:37] + "..."
            text = f"{det_class} | {doc_show} | {score:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        # putText 背景条
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(0, y1 - th - baseline - 3)
        cv2.rectangle(vis, (x1, ty), (x1 + tw + 2, ty + th + baseline + 2), color, -1)
        cv2.putText(
            vis,
            text,
            (x1 + 1, ty + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis

