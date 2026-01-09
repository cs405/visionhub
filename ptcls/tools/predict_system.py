"""System predict script (PyTorch port of visionhub deploy/python/predict_system.py).

当前实现：
- 不启用 det（只对整图做检索）
- rec embedding + faiss search

后续：可以接入你仓库现成的 YOLO 检测（ultralytics）作为 det_predictor。

使用：
python visionhub/tools/predict_system.py -c path/to/config.yaml
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.utils import config as cfg_mod
from ptcls.utils import logger
from ptcls.pipeline.shitu import SystemPredictor
from ptcls.utils.visualize import draw_shitu_results


def parse_args():
    parser = argparse.ArgumentParser(description="visionhub system predictor")
    parser.add_argument("-c", "--config", required=True, help="config yaml")
    parser.add_argument("-o", "--override", nargs="*", help="override opts")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = cfg_mod.get_config(args.config, overrides=args.override, show=True)

    log_dir = cfg.Global.get("save_inference_dir", "./output")
    os.makedirs(log_dir, exist_ok=True)
    logger.init_logger(log_file=os.path.join(log_dir, "predict_system.log"), log_level="INFO")

    infer_img = cfg.Global.get("infer_imgs")
    if not infer_img:
        raise ValueError("Global.infer_imgs is required")

    img = cv2.imread(infer_img)
    if img is None:
        raise FileNotFoundError(infer_img)
    img = img[:, :, ::-1]  # BGR->RGB

    predictor = SystemPredictor(cfg)
    out = predictor.predict(img)
    print(out)

    # visualize
    save_path = cfg.Global.get("save_visual_path")
    if not save_path:
        save_path = os.path.join(log_dir, "system_vis.jpg")

    score_thres = float(cfg.IndexProcess.get("score_thres", 0.0))
    vis = draw_shitu_results(img, out, score_thres=score_thres)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, vis)
    print(f"visualized result saved to: {save_path}")


if __name__ == "__main__":
    main()

