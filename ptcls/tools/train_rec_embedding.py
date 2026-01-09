"""Train embedding model for retrieval (minimal baseline).

目标：提升检索效果
- backbone: ResNet18/50
- head: EmbeddingHead
- loss: SupConLoss (supervised contrastive)

该脚本目前是一个“最小可用”的训练入口，先把：
- 数据格式
- 前向输出（feature/embedding）
- loss
- checkpoint 保存
跑顺。

Usage (example):
python visionhub/tools/train_rec_embedding.py \
  -c visionhub/configs/shitu/rec_faiss_demo.yaml \
  --train_root ./label_images \
  --save_dir ./visionhub/output_rec_train \
  --epochs 5

注意：
- label_images 的子目录名会被当作类别。
- 训练得到的权重保存为 .pth，可在 cfg.Global.rec_inference_model_dir 指向该权重用于检索。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch import build_model
from ptcls.data.datasets.image_folder_pairs import ImageFolderDataset
from ptcls.data.datasets.yolo_det_crop_dataset import YoloDetCropDataset
from ptcls.metric.contrastive import SupConLoss
from ptcls.utils import config as cfg_mod


def _make_collate(transform):
    """Windows-friendly collate_fn (pickleable for num_workers>0)."""

    class _Collate:
        def __init__(self, tfm):
            self.tfm = tfm

        def __call__(self, batch):
            imgs = []
            labels = []
            for item in batch:
                imgs.append(item[0])
                labels.append(item[1])
            x = torch.stack([self.tfm(im) for im in imgs], dim=0)
            y = torch.tensor(labels, dtype=torch.long)
            return x, y

    return _Collate(transform)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--train_root", default=None, help="classification folder like train_rec_data/")
    p.add_argument("--yolo_images", default=None, help="YOLO det images dir, e.g. dataset/images/train")
    p.add_argument("--yolo_labels", default=None, help="YOLO det labels dir, e.g. dataset/labels/train")
    p.add_argument("--data_yaml", default=None, help="YOLO data.yaml containing names mapping")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_path = None

    cfg = cfg_mod.get_config(args.config, overrides=None, show=False)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # dataset
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.yolo_images and args.yolo_labels:
        ds = YoloDetCropDataset(args.yolo_images, args.yolo_labels, data_yaml=args.data_yaml)
    elif args.train_root:
        ds = ImageFolderDataset(args.train_root)
    else:
        raise ValueError("Please provide either --train_root or both --yolo_images/--yolo_labels")

    collate_fn = _make_collate(tfm)
    num_workers = int(args.num_workers) if int(args.num_workers) >= 0 else 0
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)

    # model
    model = build_model(cfg)
    model.to(device)
    model.train()

    # loss/optim
    crit = SupConLoss(temperature=0.07)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        total = 0.0
        step = 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.size(0) < 2:
                continue
            out = model(x)
            emb = out.get("embedding")
            if emb is None:
                raise RuntimeError("Model must output embedding. Please enable Arch.Head=EmbeddingHead")

            loss = crit(emb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            step += 1

        avg = total / max(step, 1)
        print(f"epoch {ep}/{args.epochs} loss={avg:.4f} steps={step}")

        ckpt_path = os.path.join(args.save_dir, f"rec_ep{ep}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("done. last checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
