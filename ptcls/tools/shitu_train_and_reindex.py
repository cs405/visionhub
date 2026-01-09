"""One-command: train (optional KD) -> update cfg -> rebuild gallery index.

目标：
- 训练得到 student 权重（baseline 或 KD）
- 自动把权重路径写回到 shitu config 的 Global.rec_inference_model_dir
- 自动调用 build_gallery 重建索引

示例：
1) baseline（不蒸馏）：
python visionhub/tools/shitu_train_and_reindex.py -c visionhub/configs/shitu/rec_faiss_demo.yaml \
  --mode baseline --yolo_images dataset/images/train --yolo_labels dataset/labels/train --data_yaml dataset/data.yaml \
  --save_dir visionhub/output_rec_train --epochs 5

2) KD 蒸馏（teacher=R50 student=R18）：
python visionhub/tools/shitu_train_and_reindex.py -c visionhub/configs/shitu/rec_faiss_demo.yaml \
  --mode kd --yolo_images dataset/images/train --yolo_labels dataset/labels/train --data_yaml dataset/data.yaml \
  --save_dir visionhub/output_rec_kd --epochs 5

你也可以用 --train_root 指向“分类文件夹结构”的训练集。
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
    p.add_argument("--mode", choices=["baseline", "kd"], default="kd")

    p.add_argument("--train_root", default=None)
    p.add_argument("--yolo_images", default=None)
    p.add_argument("--yolo_labels", default=None)
    p.add_argument("--data_yaml", default=None)

    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    # Prefer GPU by default (will fall back inside training script if unavailable)
    p.add_argument("--device", default="cuda")

    # kd weights
    p.add_argument("--w_kd_embed", type=float, default=1.0)
    p.add_argument("--w_supcon", type=float, default=0.2)
    p.add_argument("--w_kd_logits", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=4.0)

    # retrieval training extras
    p.add_argument("--use_pk", action="store_true")
    p.add_argument("--P", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--w_triplet", type=float, default=1.0)
    p.add_argument("--triplet_margin", type=float, default=0.2)

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

    # 1) train
    if args.mode == "baseline":
        script = os.path.join(root, "tools", "train_rec_embedding.py")
        cmd = [py, script, "-c", os.path.abspath(args.config), "--save_dir", os.path.abspath(args.save_dir), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(args.lr), "--device", args.device]
        if args.yolo_images and args.yolo_labels:
            cmd += ["--yolo_images", os.path.abspath(args.yolo_images), "--yolo_labels", os.path.abspath(args.yolo_labels)]
            if args.data_yaml:
                cmd += ["--data_yaml", os.path.abspath(args.data_yaml)]
        elif args.train_root:
            cmd += ["--train_root", os.path.abspath(args.train_root)]
        else:
            raise ValueError("provide --train_root or --yolo_images/--yolo_labels")
        _run(cmd)
        # last ckpt
        ckpt = os.path.join(os.path.abspath(args.save_dir), f"rec_ep{args.epochs}.pth")

    else:
        script = os.path.join(root, "tools", "train_rec_kd.py")
        cmd = [py, script, "--save_dir", os.path.abspath(args.save_dir), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(args.lr), "--device", args.device]
        if args.yolo_images and args.yolo_labels:
            cmd += ["--yolo_images", os.path.abspath(args.yolo_images), "--yolo_labels", os.path.abspath(args.yolo_labels)]
            if args.data_yaml:
                cmd += ["--data_yaml", os.path.abspath(args.data_yaml)]
        elif args.train_root:
            cmd += ["--train_root", os.path.abspath(args.train_root)]
        else:
            raise ValueError("provide --train_root or --yolo_images/--yolo_labels")

        cmd += ["--w_kd_embed", str(args.w_kd_embed), "--w_supcon", str(args.w_supcon), "--w_kd_logits", str(args.w_kd_logits), "--temperature", str(args.temperature)]
        if args.use_pk:
            cmd += ["--use_pk", "--P", str(args.P), "--K", str(args.K)]
        cmd += ["--w_triplet", str(args.w_triplet), "--triplet_margin", str(args.triplet_margin)]
        _run(cmd)
        ckpt = os.path.join(os.path.abspath(args.save_dir), f"student_ep{args.epochs}.pth")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"expected checkpoint not found: {ckpt}")

    # 2) patch config
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("Global", {})
    cfg["Global"]["rec_inference_model_dir"] = ckpt

    # optional: keep a backup
    backup = cfg_path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(backup, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[OK] updated config rec_inference_model_dir -> {ckpt}")
    print(f"[OK] backup saved: {backup}")

    # 3) rebuild gallery
    build_script = os.path.join(root, "tools", "build_gallery.py")
    _run([py, build_script, "-c", cfg_path])

    print("[DONE] train + update cfg + rebuild index")


if __name__ == "__main__":
    main()

