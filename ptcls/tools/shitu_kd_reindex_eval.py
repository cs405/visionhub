"""One-command end-to-end: KD train -> write cfg -> rebuild index -> strict retrieval eval.

This is the stable "one click" entry for the YOLO+Shitu retrieval pipeline.
It wraps:
- tools/train_rec_kd.py
- tools/build_gallery.py
- tools/eval_retrieval.py

Design goals:
- Windows-friendly (no shell tricks)
- deterministic argument passing
- strong defaults: cuda + amp + PKSampler + Triplet

Example:
python visionhub/tools/shitu_kd_reindex_eval.py -c visionhub/configs/shitu/rec_faiss_demo.yaml \
  --data_yaml dataset/data.yaml \
  --yolo_train_images dataset/train --yolo_train_labels dataset/train \
  --gallery_images dataset/images/val --gallery_labels dataset/labels/val \
  --query_images dataset/images/test --query_labels dataset/labels/test \
  --save_dir visionhub/output_rec_kd \
  --epochs 5 --batch_size 32 --device cuda --amp \
  --use_pk --P 8 --K 4 --w_triplet 1.0 --triplet_margin 0.2 \
  --strict_image_split --exclude_same_image --save_eval_dir visionhub/output_rec_kd/eval

Notes:
- YOLO train data can be either:
  1) images_dir=dataset/images/train, labels_dir=dataset/labels/train
  2) split_dir=dataset/train with subfolders images/ and labels/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)

    # train data
    p.add_argument("--yolo_train_images", required=True)
    p.add_argument("--yolo_train_labels", required=True)
    p.add_argument("--data_yaml", default=None)

    # eval data
    p.add_argument("--gallery_images", required=True)
    p.add_argument("--gallery_labels", required=True)
    p.add_argument("--query_images", required=True)
    p.add_argument("--query_labels", required=True)

    # train args
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    # KD + retrieval loss
    p.add_argument("--w_kd_embed", type=float, default=1.0)
    p.add_argument("--w_supcon", type=float, default=0.2)
    p.add_argument("--w_kd_logits", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=4.0)

    p.add_argument("--use_pk", action="store_true")
    p.add_argument("--P", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--w_triplet", type=float, default=1.0)
    p.add_argument("--triplet_margin", type=float, default=0.2)

    # eval strict
    p.add_argument("--strict_image_split", action="store_true")
    p.add_argument("--strict_pick", choices=["first", "random", "max"], default="max")
    p.add_argument("--exclude_same_image", action="store_true")
    p.add_argument("--save_eval_dir", default=None)
    p.add_argument("--topk", type=int, default=5)

    return p.parse_args()


def _run(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py = sys.executable

    # 1) KD train
    train_script = os.path.join(root, "tools", "train_rec_kd.py")
    cmd = [
        py,
        train_script,
        "--save_dir",
        os.path.abspath(args.save_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--yolo_images",
        os.path.abspath(args.yolo_train_images),
        "--yolo_labels",
        os.path.abspath(args.yolo_train_labels),
        "--w_kd_embed",
        str(args.w_kd_embed),
        "--w_supcon",
        str(args.w_supcon),
        "--w_kd_logits",
        str(args.w_kd_logits),
        "--temperature",
        str(args.temperature),
        "--w_triplet",
        str(args.w_triplet),
        "--triplet_margin",
        str(args.triplet_margin),
        "--eval_every",
        "0",
    ]
    if args.data_yaml:
        cmd += ["--data_yaml", os.path.abspath(args.data_yaml)]
    if args.amp:
        cmd += ["--amp"]
    if args.use_pk:
        cmd += ["--use_pk", "--P", str(args.P), "--K", str(args.K)]
    _run(cmd)

    ckpt = os.path.join(os.path.abspath(args.save_dir), f"student_ep{args.epochs}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"expected checkpoint not found: {ckpt}")

    # 2) patch config (write rec_inference_model_dir)
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("Global", {})
    cfg["Global"]["rec_inference_model_dir"] = ckpt

    backup = cfg_path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(backup, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[OK] updated config rec_inference_model_dir -> {ckpt}")
    print(f"[OK] backup saved: {backup}")

    # 3) rebuild gallery index (this uses label_images by config IndexProcess.image_root)
    build_script = os.path.join(root, "tools", "build_gallery.py")
    _run([py, build_script, "-c", cfg_path])

    # 4) strict retrieval eval (yolo crop)
    eval_script = os.path.join(root, "tools", "eval_retrieval.py")
    ecmd = [
        py,
        eval_script,
        "-c",
        cfg_path,
        "--gallery_images",
        os.path.abspath(args.gallery_images),
        "--gallery_labels",
        os.path.abspath(args.gallery_labels),
        "--query_images",
        os.path.abspath(args.query_images),
        "--query_labels",
        os.path.abspath(args.query_labels),
        "--topk",
        str(args.topk),
    ]
    if args.data_yaml:
        ecmd += ["--data_yaml", os.path.abspath(args.data_yaml)]
    if args.strict_image_split:
        ecmd += ["--strict_image_split", "--strict_pick", args.strict_pick]
    if args.exclude_same_image:
        ecmd += ["--exclude_same_image"]
    if args.save_eval_dir:
        ecmd += ["--save_dir", os.path.abspath(args.save_eval_dir)]

    _run(ecmd)

    print("[DONE] KD train + write cfg + rebuild index + eval")


if __name__ == "__main__":
    main()

