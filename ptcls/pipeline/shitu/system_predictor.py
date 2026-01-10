"""ptcls.pipeline.shitu.system_predictor

Professional SystemPredictor for complete image search pipeline:
- Optional detection module (interface ready)
- Recognition embedding
- Faiss search
- Simple NMS deduplication

Note:
- Currently supports "no detector, full-image retrieval" mode.
- YOLO/ultralytics integration available for detection capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ...rec.predictor import RecPredictor
from ...index.faiss_index import FaissIndexManager
from ...det import YOLODetPredictor


class SystemPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.rec_predictor = RecPredictor(config)

        # Optional YOLO detector
        g = self.config.get("Global", {})
        det_model_path = g.get("det_model_path")
        det_conf = g.get("det_conf", 0.25)
        det_iou = g.get("det_iou", 0.7)
        det_device = g.get("device", "cpu")
        self.det_predictor = None
        if det_model_path:
            self.det_predictor = YOLODetPredictor(
                model_path=det_model_path,
                device=str(det_device),
                conf=float(det_conf),
                iou=float(det_iou),
            )

        if "IndexProcess" not in config:
            raise KeyError("IndexProcess not found")

        idx_cfg = config["IndexProcess"]
        self.return_k = int(idx_cfg.get("return_k", 5))
        self.dist_type = idx_cfg.get("dist_type", "IP")

        index_dir = idx_cfg["index_dir"]
        manager = FaissIndexManager(index_dir=index_dir, dist_type=self.dist_type)
        self.index, self.id_map = manager.load()
        self.manager = manager

        self.debug = bool(self.config.get("Global", {}).get("debug", False))

    def _append_self(self, results, shape):
        results.append(
            {
                "bbox": np.array([0, 0, shape[1], shape[0]]),
                "score": 1.0,
                "class_id": -1,
                "class_name": "full_image",
            }
        )
        return results

    def _nms(self, results: List[Dict[str, Any]], thresh: float = 0.1):
        if not results:
            return []
        x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
        y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
        x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
        y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
        scores = np.array([r.get("rec_scores", 0.0) for r in results]).astype("float32")

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return [results[i] for i in keep]

    def predict(self, img: np.ndarray):
        # img: RGB ndarray
        output = []

        det_results = []
        if self.det_predictor is not None:
            det_results = self.det_predictor.predict(img)

        # 全图作为一个候选（提升召回，与 visionhub 一致）
        det_results = self._append_self(det_results, img.shape)

        if self.debug:
            print(f"[DEBUG] det_results={len(det_results)} (including full_image)")

        for det in det_results:
            xmin, ymin, xmax, ymax = det["bbox"].astype(int)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.shape[1], xmax)
            ymax = min(img.shape[0], ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img[ymin:ymax, xmin:xmax, :].copy()

            feat = self.rec_predictor.predict(crop)
            scores, docs = self.manager.search(self.index, feat, self.return_k)

            top1_score = float(scores[0][0])
            top1_id = int(docs[0][0])
            top1_doc = None
            if top1_id >= 0 and top1_id in self.id_map:
                top1_doc = str(self.id_map[top1_id])

            if self.debug:
                print(
                    f"[DEBUG] det={det.get('class_name')} score={det.get('score', 1.0):.3f} "
                    f"bbox={[int(xmin), int(ymin), int(xmax), int(ymax)]} top1_score={top1_score:.4f} top1_id={top1_id} top1_doc={top1_doc}"
                )

            preds = {
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "det_score": float(det.get("score", 1.0)),
                "det_class": det.get("class_name"),
                "rec_scores": top1_score,
                "rec_docs": None,
            }

            if top1_doc is not None:
                preds["rec_docs"] = top1_doc.split()[-1]

            # score threshold
            score_thres = float(self.config["IndexProcess"].get("score_thres", -1e9))
            if self.dist_type == "hamming":
                radius = float(self.config["IndexProcess"].get("hamming_radius", 0))
                if preds["rec_scores"] <= radius:
                    output.append(preds)
            else:
                if preds["rec_scores"] >= score_thres:
                    output.append(preds)

        nms_th = float(self.config.get("Global", {}).get("rec_nms_thresold", 0.1))
        output = self._nms(output, thresh=nms_th)
        return output
