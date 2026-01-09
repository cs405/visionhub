"""Build gallery index (PyTorch port of visionhub deploy/python/build_gallery.py).

功能：
- 读取 IndexProcess.data_file（图片列表文件）
- 批量提取 embedding
- new/append/remove 方式写入 faiss index（vector.index）与 id_map.pkl

使用：
python visionhub/tools/build_gallery.py -c path/to/config.yaml
"""

import argparse
import os
import sys

# add visionhub to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.utils import config as cfg_mod
from ptcls.utils import logger
from ptcls.rec import RecPredictor
from ptcls.index import GalleryBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="visionhub build gallery index")
    parser.add_argument("-c", "--config", required=True, help="config yaml")
    parser.add_argument("-o", "--override", nargs="*", help="override opts, like A.B=xxx")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = cfg_mod.get_config(args.config, overrides=args.override, show=True)

    log_dir = cfg.Global.get("save_inference_dir", "./output")
    os.makedirs(log_dir, exist_ok=True)
    logger.init_logger(log_file=os.path.join(log_dir, "build_gallery.log"), log_level="INFO")

    rec = RecPredictor(cfg)
    builder = GalleryBuilder(cfg, rec_predictor=rec, config_path=args.config)
    builder.build()


if __name__ == "__main__":
    main()

