"""ptcls.rec.predictor

Professional RecPredictor，目标对齐 visionhub/deploy/python/predict_rec.py。

特征：
- visionhub 版本依赖 visionhub Inference Predictor/ONNX Predictor
- 内置支持 PyTorch eager 推理（torch.nn.Module + state_dict）

配置约定（尽量贴近 visionhub）：
Global:
  rec_inference_model_dir: path/to/model.pth
  device: cpu|gpu
  batch_size: 32
RecPreProcess:
  transform_ops: [...]  # 同 ptcls.data.preprocess.create_operators
RecPostProcess:
  name: None / ...      # 预留

输出：
- np.ndarray, shape [B, D]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from ..arch import build_model
from ..data.preprocess import create_operators, transform
from ..utils import logger


def _infer_device(global_cfg: dict) -> torch.device:
    dev = str(global_cfg.get("device", "gpu")).lower()
    if dev in ["gpu", "cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RecPredictor:
    """Embedding 推理器。"""

    def __init__(self, config: dict):
        self.config = config
        self.global_cfg = config.get("Global", {})

        self.device = _infer_device(self.global_cfg)

        # 1) build model
        # 这里复用 ptcls 的 build_model（目前是分类 BaseModel）。
        # 对于 rec 模型，建议 Arch.Backbone.class_num/Head 改造为 embedding 输出。
        # 当前阶段：作为“最小闭环”，直接从 backbone logits 当 embedding（后续会替换为真正的 embedding head）。
        self.model = build_model(config)
        self.model.to(self.device)
        self.model.eval()

        # 2) load weights
        model_path = self.global_cfg.get("rec_inference_model_dir")
        if model_path:
            self._load_weights(model_path)

        # 3) preprocess
        rec_pre_cfg = config.get("RecPreProcess", {})
        ops_cfg = rec_pre_cfg.get("transform_ops", [])
        self.preprocess_ops = create_operators(ops_cfg).ops if ops_cfg else []

    def _load_weights(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"rec_inference_model_dir not found: {model_path}")

        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys when load rec model: {missing[:10]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys when load rec model: {unexpected[:10]} ...")

        logger.info(f"Loaded rec model weights: {model_path}")

    @torch.no_grad()
    def predict(self, images: Union[np.ndarray, Sequence[np.ndarray]], feature_normalize: bool = True) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]

        # preprocess each image
        proc = []
        for img in images:
            x = img
            for op in self.preprocess_ops:
                x = op(x)
            # 确保 tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            proc.append(x)

        batch = torch.cat(proc, dim=0).float().to(self.device)

        out = self.model(batch)
        if isinstance(out, dict):
            feat = out.get("embedding")
            if feat is None:
                feat = out.get("logits")
        else:
            feat = out

        feat = feat.detach().float().cpu().numpy()

        if feature_normalize:
            norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
            feat = feat / norm

        return feat

