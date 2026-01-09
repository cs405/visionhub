"""Debug script: verify RecPredictor embedding output shape.

Usage:
python visionhub/tools/debug_rec_feature.py -c visionhub/configs/shitu/rec_faiss_demo.yaml -i demo.jpg
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.rec.predictor import RecPredictor
from ptcls.utils import config as cfg_mod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-i", "--image", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = cfg_mod.get_config(args.config, overrides=None, show=False)
    rec = RecPredictor(cfg)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    img = img[:, :, ::-1]

    feat = rec.predict(img)
    print("embedding shape:", feat.shape)


if __name__ == "__main__":
    main()

