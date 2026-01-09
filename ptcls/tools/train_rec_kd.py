"""Teacher-Student distillation training for retrieval embedding.

默认：Teacher=ResNet50, Student=ResNet18
- 训练目标是 student 的 embedding 更接近 teacher 的 embedding/feature
- 同时可叠加 supervised contrastive (SupCon) 来稳住类内聚类效果

数据：默认按文件夹名作为类别（复用 ImageFolderDataset）。

Usage (example):
python visionhub/tools/train_rec_kd.py \
  --train_root ./label_images \
  --save_dir ./visionhub/output_rec_kd \
  --epochs 5 \
  --device cpu

你也可以额外传 teacher/student 的配置 yaml（可选），否则使用内置默认。

输出：
- student checkpoint: student_epX.pth

接入检索：
- 把 cfg.Global.rec_inference_model_dir 指向 student_epX.pth
- 重建 gallery 索引
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARNING] tqdm not available, progress bar disabled. Install: pip install tqdm")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional

from ptcls.arch import build_model
from ptcls.data.datasets.image_folder_pairs import ImageFolderDataset
from ptcls.data.datasets.yolo_det_crop_dataset import YoloDetCropDataset
from ptcls.data.samplers.pk_sampler import PKBatchSampler
from ptcls.distill.losses import CosineEmbeddingDistillLoss, KLDivergenceLoss, MSEFeatLoss
from ptcls.metric.contrastive import SupConLoss
from ptcls.metric.triplet import BatchHardTripletLoss
from ptcls.metric.retrieval_eval import map_at_k, recall_at_k
from ptcls.utils import config as cfg_mod

import copy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default=None)

    # NOTE: for YOLO crop dataset we accept:
    # 1) --yolo_images dataset/train/images --yolo_labels dataset/train/labels
    # 2) OR (recommended) pass split root once:
    #    --yolo_images dataset/train --yolo_labels dataset/train
    # The dataset class will auto-expand to images/ and labels/.
    p.add_argument("--yolo_images", default=None)
    p.add_argument("--yolo_labels", default=None)
    p.add_argument("--data_yaml", default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    # prefer GPU by default
    p.add_argument("--device", default="cuda")

    # dataloader perf
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", help="pin memory when using cuda")

    # amp
    p.add_argument("--amp", action="store_true", help="use torch.cuda.amp for faster training on GPU")

    p.add_argument("--teacher_config", default=None)
    p.add_argument("--teacher_ckpt", default=None)
    p.add_argument("--student_config", default=None)
    p.add_argument("--student_ckpt", default=None)

    # loss weights
    p.add_argument("--w_kd_embed", type=float, default=1.0)
    p.add_argument("--w_supcon", type=float, default=0.2)
    p.add_argument("--w_kd_logits", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=4.0)

    # PK sampler
    p.add_argument("--P", type=int, default=8, help="num classes per batch")
    p.add_argument("--K", type=int, default=4, help="samples per class")
    p.add_argument("--use_pk", action="store_true", help="enable PK sampler (recommended)")

    # retrieval loss
    p.add_argument("--w_triplet", type=float, default=1.0)
    p.add_argument("--triplet_margin", type=float, default=0.2)

    # metric learning losses (additional)
    p.add_argument("--w_circle", type=float, default=0.0)
    p.add_argument("--circle_m", type=float, default=0.25)
    p.add_argument("--circle_gamma", type=float, default=64.0)

    p.add_argument("--w_arcface", type=float, default=0.0, help="if >0, enable ArcFace CE loss")
    p.add_argument("--arcface_s", type=float, default=64.0)
    p.add_argument("--arcface_m", type=float, default=0.5)

    # ImageNet pretrained (torchvision) for teacher/student
    p.add_argument("--teacher_pretrained", action="store_true")
    p.add_argument("--student_pretrained", action="store_true")

    # eval
    p.add_argument("--eval_every", type=int, default=1)

    # validation (YOLO crop) for early-stop/best
    p.add_argument("--val_yolo_images", default=None)
    p.add_argument("--val_yolo_labels", default=None)
    p.add_argument("--monitor", choices=["map@10", "recall@1"], default="map@10")
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--patience", type=int, default=0, help="early stop patience; 0 disables")

    # teacher backbone option: use torchvision native model as teacher (strict pretrained)
    p.add_argument(
        "--teacher_torchvision",
        action="store_true",
        help="use torchvision resnet as teacher backbone (weights strictly aligned, recommended)",
    )

    # skip steps when loss is NaN/Inf
    p.add_argument("--skip_non_finite", action="store_true", help="skip steps when loss is NaN/Inf")

    # gradient clipping
    p.add_argument("--grad_clip", type=float, default=0.0, help="clip grad norm; 0 disables")
    p.add_argument("--max_skip_ratio", type=float, default=0.5, help="if skipped_steps/total_steps > ratio, fallback")
    p.add_argument("--fallback_no_amp", action="store_true", help="auto disable amp when too many NaN")

    # strict val eval (align with tools/eval_retrieval.py)
    p.add_argument("--val_gallery_images", default=None)
    p.add_argument("--val_gallery_labels", default=None)
    p.add_argument("--val_query_images", default=None)
    p.add_argument("--val_query_labels", default=None)
    p.add_argument("--val_strict_image_split", action="store_true")
    p.add_argument("--val_strict_pick", choices=["first", "random", "max"], default="max")
    p.add_argument("--val_exclude_same_image", action="store_true")
    p.add_argument("--val_map_k", type=int, default=10)
    p.add_argument("--val_topk", type=int, default=10)

    # warmup strategy (stability)
    p.add_argument("--warmup_epochs", type=int, default=3, help="epochs to warm up before enabling circle/arcface")
    p.add_argument(
        "--warmup_mode",
        choices=["linear", "cosine", "step"],
        default="linear",
        help="how to ramp metric losses after warmup",
    )
    p.add_argument(
        "--warmup_disable_supcon",
        action="store_true",
        help="during warmup, force w_supcon=0 to stabilize",
    )
    p.add_argument(
        "--warmup_circle_gamma",
        type=float,
        default=None,
        help="optional: use a smaller circle_gamma during warmup (e.g., 32), then restore",
    )

    # LR scheduler
    p.add_argument("--scheduler", choices=["none", "step", "cosine", "exponential"], default="cosine", help="learning rate scheduler type")
    p.add_argument("--lr_step_size", type=int, default=10, help="epoch interval for step scheduler")
    p.add_argument("--lr_gamma", type=float, default=0.1, help="multiplicative factor for step/exponential scheduler")
    p.add_argument("--lr_min", type=float, default=1e-6, help="minimum learning rate for cosine scheduler")

    return p.parse_args()


def _default_teacher_student_cfg(base_cfg_path: str):
    """Build default teacher/student configs from a base yaml.

    Be robust to both AttrDict and plain dict shapes.
    """
    base = cfg_mod.get_config(base_cfg_path, overrides=None, show=False)

    # Work on plain dict copies, then convert back to AttrDict.
    base_plain = copy.deepcopy(dict(base))
    teacher_plain = copy.deepcopy(base_plain)
    student_plain = copy.deepcopy(base_plain)

    def ensure(d: dict, path: list[str], default_factory):
        cur = d
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        last = path[-1]
        if last not in cur or cur[last] is None:
            cur[last] = default_factory()
        return cur[last]

    # Ensure required structure exists
    ensure(teacher_plain, ["Arch"], dict)
    ensure(teacher_plain, ["Arch", "Backbone"], dict)
    ensure(teacher_plain, ["Arch", "Head"], dict)

    ensure(student_plain, ["Arch"], dict)
    ensure(student_plain, ["Arch", "Backbone"], dict)
    ensure(student_plain, ["Arch", "Head"], dict)

    # embedding_size priority: IndexProcess.embedding_size -> Arch.Head.embedding_size -> 512
    emb_size = None
    try:
        emb_size = int(getattr(getattr(base, "IndexProcess", None), "embedding_size", None))
    except Exception:
        emb_size = None
    if emb_size is None:
        try:
            emb_size = int(getattr(getattr(getattr(base, "Arch", None), "Head", None), "embedding_size", None))
        except Exception:
            emb_size = None
    if emb_size is None:
        emb_size = 512

    # Apply defaults
    teacher_plain["Arch"]["Backbone"]["name"] = "ResNet50"
    teacher_plain["Arch"]["Backbone"]["class_num"] = 1000
    teacher_plain["Arch"]["Head"].setdefault("name", "EmbeddingHead")
    teacher_plain["Arch"]["Head"]["embedding_size"] = emb_size

    student_plain["Arch"]["Backbone"]["name"] = "ResNet18"
    student_plain["Arch"]["Backbone"]["class_num"] = 1000
    student_plain["Arch"]["Head"].setdefault("name", "EmbeddingHead")
    student_plain["Arch"]["Head"]["embedding_size"] = emb_size

    teacher = cfg_mod.create_attr_dict(teacher_plain)
    student = cfg_mod.create_attr_dict(student_plain)

    return teacher, student


def _make_collate(transform):
    """Create a pickle-friendly collate function.

    On Windows, DataLoader(num_workers>0) requires collate_fn to be pickleable,
    so we return a top-level callable (not a local function).
    """

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


def _load_torchvision_pretrained(backbone, name: str):
    """Best-effort load torchvision ImageNet weights into our custom ResNet.

    Our ResNet implementation is visionhub-style and doesn't share exact key names with torchvision,
    so we do a safe, partial load on matching shapes.

    This gives some benefit (especially early layers) without being brittle.
    """
    try:
        import torchvision

        if name.lower() == "resnet18":
            tv = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif name.lower() == "resnet50":
            tv = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            return 0

        tv_state = tv.state_dict()
        own_state = backbone.state_dict()
        loaded = 0
        for k, v in tv_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
        backbone.load_state_dict(own_state, strict=False)
        return loaded
    except Exception:
        return 0


def _build_val_loader(args, tfm, device):
    if not (args.val_yolo_images and args.val_yolo_labels):
        return None

    ds = YoloDetCropDataset(args.val_yolo_images, args.val_yolo_labels, data_yaml=args.data_yaml)
    labels = [int(getattr(s, "label", getattr(s, "cls_id"))) for s in getattr(ds, "samples")]
    dl = DataLoader(
        ds,
        batch_size=min(int(args.batch_size), 64),
        shuffle=False,
        num_workers=int(args.num_workers) if int(args.num_workers) >= 0 else 0,
        collate_fn=_make_collate(tfm),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dl


@torch.no_grad()
def _eval_loader(model, dl, device, autocast_ctx, max_batches: int = 50):
    import numpy as np

    if dl is None:
        return None

    model.eval()
    embs = []
    labs = []
    nb = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            out = model(x)
            e = out.get("embedding") if isinstance(out, dict) else out
        if e is None:
            continue
        embs.append(e.detach().float().cpu().numpy())
        labs.append(y.numpy())
        nb += 1
        if max_batches and nb >= max_batches:
            break

    if not embs:
        return None

    E = np.concatenate(embs, axis=0)
    L = np.concatenate(labs, axis=0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = E @ E.T  # use itself as gallery for quick val proxy
    rec = recall_at_k(sim, L, L, ks=(1, 5, 10))
    m10 = map_at_k(sim, L, L, k=10)
    return {"recall@1": float(rec[1]), "map@10": float(m10)}


@torch.no_grad()
def _eval_loader_strict(model, dl, device, autocast_ctx, *, exclude_self: bool = True, topk: int = 10):
    """Strict-ish retrieval eval without faiss.

    We split each batch into query/gallery halves and compute retrieval metrics.
    This avoids self-similarity causing unrealistically high early metrics.

    This is still a proxy, but much closer to your final eval than pure self-gallery.
    """
    import numpy as np

    if dl is None:
        return None

    model.eval()
    E_all = []
    L_all = []
    for x, y in dl:
        if x.size(0) < 4:
            continue
        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            out = model(x)
            e = out.get("embedding") if isinstance(out, dict) else out
        if e is None:
            continue
        E_all.append(e.detach().float().cpu().numpy())
        L_all.append(y.numpy())

    if not E_all:
        return None

    E = np.concatenate(E_all, axis=0)
    L = np.concatenate(L_all, axis=0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # query/gallery split by index parity
    q_idx = np.arange(len(L)) % 2 == 0
    g_idx = ~q_idx
    if q_idx.sum() < 2 or g_idx.sum() < 2:
        return None

    Q = E[q_idx]
    G = E[g_idx]
    q_lab = L[q_idx]
    g_lab = L[g_idx]

    sim = Q @ G.T

    # compute recall@1 and mAP@10
    # for each query, rank gallery by similarity
    idx = np.argsort(-sim, axis=1)
    top = idx[:, :topk]
    rel = (g_lab[top] == q_lab[:, None]).astype(np.float32)

    r1 = float((rel[:, 0] > 0).mean())

    # AP@k
    prec = np.cumsum(rel, axis=1) / (np.arange(topk)[None, :] + 1)
    ap = (prec * rel).sum(axis=1) / np.maximum(rel.sum(axis=1), 1.0)
    map10 = float(ap.mean())

    return {"recall@1": r1, "map@10": map10}


class _TvTeacher(torch.nn.Module):
    """Torchvision teacher that outputs {'embedding','logits'} compatible with our training loop."""

    def __init__(self, name: str = "resnet50", embedding_size: int = 512):
        super().__init__()
        import torchvision
        import torch.nn as nn

        name = name.lower()
        if name == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        else:
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048

        # split backbone
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # upto avgpool
        self.pool_flatten = nn.Flatten(1)
        self.fc = m.fc  # logits head (1000)

        # embedding head: linear + bn (no l2norm here; student has l2norm already)
        from ptcls.arch.head.embedding_head import EmbeddingHead

        self.emb = EmbeddingHead(in_dim=feat_dim, embedding_size=int(embedding_size), with_l2norm=True)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool_flatten(feat)
        logits = self.fc(feat)
        emb = self.emb(feat)
        return {"feature": feat, "logits": logits, "embedding": emb}


def _arcface_forward(arcface, emb, y):
    """Forward ArcFace head with a hard assertion (for clearer errors + static check friendliness)."""
    assert arcface is not None, "w_arcface>0 but arcface head is not initialized"
    return arcface(emb, y)


def _eval_val_by_eval_retrieval(cfg_path: str, ckpt_path: str, args, *, device: str = "cuda"):
    """Strict retrieval val eval aligned with tools/eval_retrieval.py (no faiss).

    Returns: {"recall@1": float, "map@10": float}

    Notes:
    - Uses RecPredictor to keep preprocessing/normalization consistent with inference.
    - Excludes same-image matches if args.val_exclude_same_image is enabled.
    - Supports strict_image_split (one bbox per image) with pick strategy.
    """
    import numpy as np
    import tempfile

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyYAML required for strict val eval: {e}")

    # create a temp config pointing to current checkpoint
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_plain = yaml.safe_load(f) or {}
    cfg_plain.setdefault("Global", {})
    cfg_plain["Global"]["rec_inference_model_dir"] = str(ckpt_path)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg_plain, tmp, allow_unicode=True)
    tmp.flush()
    tmp_cfg_path = tmp.name
    tmp.close()

    from ptcls.utils import config as cfg_mod
    from ptcls.rec.predictor import RecPredictor
    from ptcls.data.datasets.yolo_det_crop_dataset import YoloDetCropDataset
    from ptcls.metric.retrieval_eval import recall_at_k, map_at_k

    cfg = cfg_mod.get_config(tmp_cfg_path, overrides=None, show=False)
    rec = RecPredictor(cfg)

    g_ds = YoloDetCropDataset(args.val_gallery_images, args.val_gallery_labels, data_yaml=args.data_yaml, return_name=False)
    q_ds = YoloDetCropDataset(args.val_query_images, args.val_query_labels, data_yaml=args.data_yaml, return_name=False)

    def _extract(ds, max_n: int, strict_split: bool, strict_pick: str):
        feats = []
        labels = []
        img_ids = []

        if not strict_split:
            n = min(len(ds), int(max_n))
            for i in range(n):
                item = ds[i]
                img = item[0]
                y = int(item[1])
                image_id = str(item[-2])
                img_np = np.array(img)
                feat = rec.predict(img_np, feature_normalize=True)
                feats.append(feat[0])
                labels.append(y)
                img_ids.append(image_id)
            return np.stack(feats, axis=0), labels, img_ids

        # strict: one object per image
        import random
        from PIL import Image as _PILImage

        by_img = {}
        for s in ds.samples:
            by_img.setdefault(s.image_id, []).append(s)

        image_ids = sorted(by_img.keys())[: int(max_n)]
        for image_id in image_ids:
            items = by_img[image_id]
            if strict_pick == "random":
                s = random.choice(items)
            elif strict_pick == "max":
                def area(ss):
                    x1, y1, x2, y2 = ss.xyxy
                    return max(0, x2 - x1) * max(0, y2 - y1)

                s = max(items, key=area)
            else:
                s = items[0]

            im = _PILImage.open(s.img_path).convert("RGB")
            x1, y1, x2, y2 = s.xyxy
            crop = im.crop((x1, y1, x2, y2))
            img_np = np.array(crop)
            feat = rec.predict(img_np, feature_normalize=True)

            feats.append(feat[0])
            labels.append(int(s.cls_id))
            img_ids.append(str(image_id))

        return np.stack(feats, axis=0), labels, img_ids

    g_feat, g_lab, g_ids = _extract(
        g_ds,
        max_n=getattr(args, "max_gallery", 5000),
        strict_split=bool(getattr(args, "val_strict_image_split", False)),
        strict_pick=str(getattr(args, "val_strict_pick", "max")),
    )
    q_feat, q_lab, q_ids = _extract(
        q_ds,
        max_n=getattr(args, "max_query", 2000),
        strict_split=bool(getattr(args, "val_strict_image_split", False)),
        strict_pick=str(getattr(args, "val_strict_pick", "max")),
    )

    g_feat = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-12)
    q_feat = q_feat / (np.linalg.norm(q_feat, axis=1, keepdims=True) + 1e-12)
    sim = q_feat @ g_feat.T

    if bool(getattr(args, "val_exclude_same_image", False)):
        q_ids = np.asarray(q_ids, dtype=object)
        g_ids = np.asarray(g_ids, dtype=object)
        same = q_ids[:, None] == g_ids[None, :]
        sim = sim.copy()
        sim[same] = -1e9

    r = recall_at_k(sim, q_lab, g_lab, ks=(1, 5, 10))
    mk = int(getattr(args, "val_map_k", 10))
    m10 = map_at_k(sim, q_lab, g_lab, k=mk)

    # ensure temp file cleanup
    # (best-effort)
    try:
        import os as _os

        if tmp_cfg_path and _os.path.exists(tmp_cfg_path):
            _os.remove(tmp_cfg_path)
    except Exception:
        pass
    return {"recall@1": float(r[1]), "map@10": float(m10)}


# IMPORTANT: main must be defined after helper functions

def _ramp(ep: int, warmup_epochs: int, total_epochs: int, mode: str) -> float:
    """Warmup ramp factor in [0,1] for enabling metric losses.

    - ep is 1-based
    - if ep <= warmup_epochs: 0
    - else: ramp up to 1.0 until total_epochs
    """
    if warmup_epochs <= 0:
        return 1.0
    if ep <= warmup_epochs:
        return 0.0

    denom = max(1, total_epochs - warmup_epochs)
    t = min(1.0, max(0.0, (ep - warmup_epochs) / float(denom)))

    if mode == "step":
        return 1.0
    if mode == "cosine":
        import math

        return float(0.5 - 0.5 * math.cos(math.pi * t))
    return float(t)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_path = None

    # device selection: prefer cuda when available
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] device={device} (torch.cuda.is_available()={torch.cuda.is_available()})", flush=True)

    # dataset
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # IMPORTANT: YoloDetCropDataset expects (images_dir, labels_dir)
    # It can auto-expand if you pass split root containing images/ and labels/.
    if args.yolo_images and args.yolo_labels:
        print(f"[INFO] building YoloDetCropDataset: images={args.yolo_images} labels={args.yolo_labels} data_yaml={args.data_yaml}", flush=True)
        ds = YoloDetCropDataset(args.yolo_images, args.yolo_labels, data_yaml=args.data_yaml)
    elif args.train_root:
        print(f"[INFO] building ImageFolderDataset: train_root={args.train_root}", flush=True)
        ds = ImageFolderDataset(args.train_root)
    else:
        raise ValueError("Please provide either --train_root or both --yolo_images/--yolo_labels")

    # basic dataset stats
    try:
        n = len(ds)
    except Exception as e:
        raise RuntimeError(f"Failed to get dataset length: {e}")
    num_classes = None
    try:
        if hasattr(ds, "samples"):
            labs = [int(getattr(s, "label", getattr(s, "cls_id"))) for s in getattr(ds, "samples")]
            num_classes = len(set(labs)) if labs else 0
    except Exception:
        num_classes = None
    print(f"[INFO] dataset_size={n}" + (f" num_classes={num_classes}" if num_classes is not None else ""), flush=True)

    collate_fn = _make_collate(tfm)

    num_workers = int(args.num_workers) if int(args.num_workers) >= 0 else 0
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    # build a unified label list for sampler/num_classes inference
    dataset_labels = None

    if args.use_pk:
        # PK sampler needs a label list aligned with dataset indices
        if hasattr(ds, "samples"):
            dataset_labels = [int(getattr(s, "label", getattr(s, "cls_id"))) for s in getattr(ds, "samples")]
        else:
            dataset_labels = [int(s.cls_id) for s in getattr(ds, "samples")]
        labels = dataset_labels
        # sanity checks for PK
        if len(labels) == 0:
            raise RuntimeError("PK sampler enabled but dataset has 0 samples.")
        cls_cnt = {}
        for lb in labels:
            cls_cnt[lb] = cls_cnt.get(lb, 0) + 1
        enough_classes = sum(1 for c in cls_cnt.values() if c >= args.K)
        if enough_classes < args.P:
            raise RuntimeError(
                f"PK sampler requires at least P={args.P} classes each with >=K={args.K} samples. "
                f"But only {enough_classes} classes satisfy this in current dataset. "
                f"Try smaller --P/--K or disable --use_pk."
            )

        batch_sampler = PKBatchSampler(labels=labels, P=args.P, K=args.K, drop_last=True, seed=0)
        dl = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    else:
        if hasattr(ds, "samples"):
            dataset_labels = [int(getattr(s, "label", getattr(s, "cls_id"))) for s in getattr(ds, "samples")]
        else:
            dataset_labels = [int(s.cls_id) for s in getattr(ds, "samples")]
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
        )

    # quick dataloader smoke test (to avoid "looks stuck" cases)
    print(f"[INFO] Testing dataloader (use_pk={args.use_pk}, num_workers={num_workers})...", flush=True)
    try:
        print("[INFO] Creating dataloader iterator...", flush=True)
        _it = iter(dl)
        print("[INFO] Fetching first batch...", flush=True)
        _x0, _y0 = next(_it)
        print(f"[INFO] first_batch: x={tuple(_x0.shape)} y={tuple(_y0.shape)}", flush=True)
    except StopIteration:
        raise RuntimeError(
            "Dataloader produced 0 batches. Common causes: dataset too small, drop_last=True with too large batch, "
            "or PK sampler constraints not met."
        )
    except Exception as e:
        print(f"[ERROR] Dataloader test failed: {e}", flush=True)
        raise

    # build optional val loader
    val_dl = _build_val_loader(args, tfm, device)

    # configs
    base_cfg_path = args.student_config or args.teacher_config
    if base_cfg_path is None:
        base_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "shitu", "rec_faiss_demo.yaml"))

    # strict val eval requires explicit query/gallery
    use_strict_val = bool(args.val_gallery_images and args.val_gallery_labels and args.val_query_images and args.val_query_labels)

    if args.teacher_config:
        teacher_cfg = cfg_mod.get_config(args.teacher_config, overrides=None, show=False)
    else:
        teacher_cfg, _ = _default_teacher_student_cfg(base_cfg_path)

    if args.student_config:
        student_cfg = cfg_mod.get_config(args.student_config, overrides=None, show=False)
    else:
        _, student_cfg = _default_teacher_student_cfg(base_cfg_path)

    # models
    if args.teacher_torchvision:
        # embedding_size from cfg
        emb_dim = None
        try:
            emb_dim = int(getattr(getattr(teacher_cfg.Arch, "Head", None), "embedding_size", 512))
        except Exception:
            emb_dim = 512
        teacher = _TvTeacher(name=str(getattr(teacher_cfg.Arch.Backbone, "name", "resnet50")), embedding_size=int(emb_dim))
    else:
        teacher = build_model(teacher_cfg)

    student = build_model(student_cfg)

    # optionally load torchvision pretrained weights for backbone
    try:
        if args.teacher_pretrained and hasattr(teacher, "backbone"):
            n = _load_torchvision_pretrained(teacher.backbone, getattr(teacher_cfg.Arch.Backbone, "name", "ResNet50"))
            print(f"[INFO] teacher_pretrained loaded_keys={n}", flush=True)
        if args.student_pretrained and hasattr(student, "backbone"):
            n = _load_torchvision_pretrained(student.backbone, getattr(student_cfg.Arch.Backbone, "name", "ResNet18"))
            print(f"[INFO] student_pretrained loaded_keys={n}", flush=True)
    except Exception:
        pass

    teacher.to(device).eval()
    student.to(device).train()

    # load ckpt
    def load_ckpt(model, path):
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    load_ckpt(teacher, args.teacher_ckpt)
    load_ckpt(student, args.student_ckpt)

    # losses
    supcon = SupConLoss(temperature=0.07)
    kd_embed = CosineEmbeddingDistillLoss()
    kd_logits = KLDivergenceLoss(temperature=args.temperature)
    triplet = BatchHardTripletLoss(margin=args.triplet_margin)

    # metric learning losses
    from ptcls.loss import CircleLoss, ArcFaceHead

    circle = CircleLoss(m=args.circle_m, gamma=args.circle_gamma)
    arcface = None  # type: Optional[ArcFaceHead]
    if args.w_arcface > 0:
        # number of classes from YOLO data.yaml if provided; else fallback to max label + 1
        num_classes_arc = None
        try:
            if args.data_yaml:
                from ptcls.utils.yolo_data import load_yolo_data_yaml

                id2name, _ = load_yolo_data_yaml(args.data_yaml)
                num_classes_arc = max(id2name.keys()) + 1 if id2name else None
        except Exception:
            num_classes_arc = None

        if num_classes_arc is None:
            # fallback: infer from dataset labels
            try:
                if dataset_labels:
                    num_classes_arc = int(max(dataset_labels) + 1)
            except Exception:
                num_classes_arc = None
        if num_classes_arc is None:
            num_classes_arc = 1000

        # NOTE: ArcFace input dim should match embedding dim
        emb_dim = None
        try:
            emb_dim = int(getattr(getattr(student_cfg.Arch, "Head", None), "embedding_size", None))
        except Exception:
            emb_dim = None
        if emb_dim is None:
            emb_dim = 512

        arcface = ArcFaceHead(in_features=int(emb_dim), num_classes=int(num_classes_arc), s=args.arcface_s, m=args.arcface_m).to(device)

    # add arcface parameters into optimizer if enabled
    params = list(student.parameters())
    if arcface is not None:
        params += list(arcface.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    # LR Scheduler
    scheduler = None
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    elif args.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)

    ce = torch.nn.CrossEntropyLoss()

    # amp (new API first, fallback to old for older torch)
    try:
        from torch.amp import GradScaler, autocast

        scaler = GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
        _autocast = lambda: autocast("cuda", enabled=(args.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
        _autocast = lambda: torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda"))

    amp_enabled = bool(args.amp and device.type == "cuda")

    def _autocast_dyn():
        return _autocast() if amp_enabled else torch.autocast("cuda", enabled=False)

    best_score = float("-inf")
    best_path = None
    bad_epochs = 0

    nan_skipped = 0

    # Keep original target weights for warmup scheduling
    target_w_circle = float(args.w_circle)
    target_w_arcface = float(args.w_arcface)
    target_w_supcon = float(args.w_supcon)
    target_circle_gamma = float(args.circle_gamma)

    print("\n" + "="*80, flush=True)
    print("TRAINING START", flush=True)
    print("="*80, flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size if not args.use_pk else f'PK: {args.P}x{args.K}={args.P*args.K}'}", flush=True)
    print(f"Device: {device} (AMP: {amp_enabled})", flush=True)
    print(f"Loss: Triplet({args.w_triplet}) + Circle({args.w_circle}) + KD({args.w_kd_embed}) + SupCon({args.w_supcon})", flush=True)
    print(f"Warmup: {args.warmup_epochs} epochs ({args.warmup_mode} mode)", flush=True)
    print(f"Monitor: {args.monitor} (patience={args.patience})", flush=True)
    print("="*80 + "\n", flush=True)

    for ep in range(1, args.epochs + 1):
        # warmup schedule: only KD/Triplet during warmup, then ramp metric losses
        ramp = _ramp(ep, int(args.warmup_epochs), int(args.epochs), str(args.warmup_mode))

        w_circle_ep = target_w_circle * ramp
        w_arcface_ep = target_w_arcface * ramp

        if args.warmup_disable_supcon and ep <= int(args.warmup_epochs):
            w_supcon_ep = 0.0
        else:
            w_supcon_ep = target_w_supcon

        # optional smaller gamma during warmup (helps prevent overflow with AMP)
        if args.warmup_circle_gamma is not None and ep <= int(args.warmup_epochs):
            circle.gamma = float(args.warmup_circle_gamma)
        else:
            circle.gamma = float(target_circle_gamma)

        # Epoch header (YOLO style)
        epoch_start_time = time.time()
        print(f"\n{'='*100}")
        print(f"Epoch {ep}/{args.epochs}")
        print(f"{'='*100}")
        print(f"Ramp: {ramp:.3f} | SupCon: {w_supcon_ep:.3f} | Circle: {w_circle_ep:.3f} | ArcFace: {w_arcface_ep:.3f} | CircleGamma: {float(circle.gamma)}")

        total = 0.0
        step = 0
        skipped_in_ep = 0
        total_in_ep = 0

        # Create progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(dl, desc=f"Epoch {ep}/{args.epochs}",
                       ncols=120,
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            pbar = dl

        for batch_idx, (x, y) in enumerate(pbar):
            total_in_ep += 1


            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # EmbeddingHead has BatchNorm1d; BN can't train on batch size 1.
            if x.size(0) < 2:
                continue

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                with _autocast_dyn():
                    t_out = teacher(x)
                    t_emb = t_out.get("embedding")
                    t_logits = t_out.get("logits")

            with _autocast_dyn():
                s_out = student(x)
                s_emb = s_out.get("embedding")
                s_logits = s_out.get("logits")

                if s_emb is None or t_emb is None:
                    raise RuntimeError("Both teacher and student must output embedding. Enable Arch.Head=EmbeddingHead")

                # extra safety normalize (teacher/student may already normalize in head)
                s_emb_n = torch.nn.functional.normalize(s_emb, dim=1)
                t_emb_n = torch.nn.functional.normalize(t_emb, dim=1)

                loss = 0.0
                if args.w_kd_embed > 0:
                    loss = loss + args.w_kd_embed * kd_embed(s_emb_n, t_emb_n)
                if w_supcon_ep > 0:
                    loss = loss + w_supcon_ep * supcon(s_emb_n, y)
                if args.w_triplet > 0:
                    loss = loss + args.w_triplet * triplet(s_emb_n, y)
                if w_circle_ep > 0:
                    loss = loss + w_circle_ep * circle(s_emb_n, y)
                if w_arcface_ep > 0:
                    logits_arc = _arcface_forward(arcface, s_emb_n, y)
                    loss = loss + w_arcface_ep * ce(logits_arc, y)
                if args.w_kd_logits > 0 and t_logits is not None and s_logits is not None:
                    loss = loss + args.w_kd_logits * kd_logits(s_logits, t_logits)

            # skip non-finite
            if args.skip_non_finite and (not torch.isfinite(loss).all()):
                nan_skipped += 1
                skipped_in_ep += 1
                continue

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))

            scaler.step(opt)
            scaler.update()

            total += float(loss.detach().item())
            step += 1

            # Update progress bar with current metrics
            if TQDM_AVAILABLE:
                current_loss = total / max(step, 1)
                gpu_mem = f"{torch.cuda.memory_reserved() / 1E9:.1f}G" if torch.cuda.is_available() else "N/A"
                pbar.set_postfix_str(f"loss={current_loss:.4f} gpu_mem={gpu_mem} skip={skipped_in_ep}")

        # Close progress bar
        if TQDM_AVAILABLE:
            pbar.close()

        avg = total / max(step, 1)
        epoch_time = time.time() - epoch_start_time

        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']

        # Epoch summary (YOLO style)
        print(f"\nEpoch {ep}/{args.epochs} Summary:")
        print(
            f"  Loss: {avg:.4f} | Steps: {step}/{total_in_ep} | Skipped: {skipped_in_ep} | "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | AMP: {amp_enabled}",
            flush=True
        )

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # If we skipped too many steps, fallback to a safer setup
        if total_in_ep > 0 and step == 0:
            # all skipped -> likely numerical explosion; make it harder to diverge
            if args.fallback_no_amp and amp_enabled:
                amp_enabled = False
                print("[FALLBACK] disabled AMP due to all steps skipped in epoch", flush=True)
            # reduce circle gamma to be safer
            if args.w_circle > 0 and args.circle_gamma > 16:
                args.circle_gamma = float(max(16.0, args.circle_gamma / 2.0))
                circle.gamma = float(args.circle_gamma)
                print(f"[FALLBACK] reduced circle_gamma -> {args.circle_gamma}", flush=True)

        elif total_in_ep > 0 and args.skip_non_finite:
            ratio = skipped_in_ep / float(total_in_ep)
            if ratio > float(args.max_skip_ratio):
                if args.fallback_no_amp and amp_enabled:
                    amp_enabled = False
                    print(f"[FALLBACK] disabled AMP due to high skip ratio={ratio:.2f}", flush=True)
                if args.w_circle > 0 and args.circle_gamma > 16:
                    args.circle_gamma = float(max(16.0, args.circle_gamma / 2.0))
                    circle.gamma = float(args.circle_gamma)
                    print(f"[FALLBACK] reduced circle_gamma -> {args.circle_gamma} (skip ratio={ratio:.2f})", flush=True)

        # simple retrieval eval on a small subset (epoch-level)
        if args.eval_every > 0 and ep % args.eval_every == 0:
            student.eval()
            with torch.no_grad():
                embs = []
                labs = []
                # take up to 512 samples for quick eval
                cnt = 0
                for x, y in dl:
                    if x.size(0) < 2:
                        continue
                    x = x.to(device, non_blocking=True)
                    with _autocast():
                        out = student(x)
                        e = out.get("embedding")
                    if e is None:
                        break
                    embs.append(e.detach().float().cpu().numpy())
                    labs.append(y.numpy())
                    cnt += x.size(0)
                    if cnt >= 512:
                        break
                if embs:
                    import numpy as np

                    E = np.concatenate(embs, axis=0)
                    L = np.concatenate(labs, axis=0)
                    # use itself as gallery for quick sanity metrics
                    sim = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)) @ (
                        E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
                    ).T
                    rec = recall_at_k(sim, L, L, ks=(1, 5, 10))
                    m10 = map_at_k(sim, L, L, k=10)
                    print(
                        f"[eval] recall@1={rec[1]:.3f} recall@5={rec[5]:.3f} recall@10={rec[10]:.3f} mAP@10={m10:.3f}"
                    )
            student.train()

        ckpt_path = os.path.join(args.save_dir, f"student_ep{ep}.pth")
        torch.save(student.state_dict(), ckpt_path)

        # val eval + best/early-stop
        if val_dl is not None and not use_strict_val:
            metrics = _eval_loader_strict(student, val_dl, device, _autocast_dyn, topk=10)
            if metrics:
                score = float(metrics.get(args.monitor, float("-inf")))
                print(f"\n  Validation Metrics:")
                print(f"    Recall@1:  {metrics['recall@1']:.4f} | Recall@5:  {metrics.get('recall@5', 0):.4f} | Recall@10: {metrics.get('recall@10', 0):.4f}")
                print(f"    mAP@10:    {metrics['map@10']:.4f} | Monitor ({args.monitor}): {score:.4f}")

                improved = score > best_score + 1e-8
                if improved:
                    best_score = score
                    bad_epochs = 0
                    if args.save_best:
                        best_path = os.path.join(args.save_dir, "student_best.pth")
                        torch.save(student.state_dict(), best_path)
                        print(f"  ✓ New best model saved! (score={best_score:.4f}) -> {best_path}")
                else:
                    bad_epochs += 1
                    print(f"  ✗ No improvement ({bad_epochs}/{args.patience})")
                    if args.patience and bad_epochs >= int(args.patience):
                        print(f"\n{'='*100}")
                        print(f"Early stopping triggered! No improvement for {bad_epochs} epochs.")
                        print(f"Best score: {best_score:.4f} (epoch {ep - bad_epochs})")
                        print(f"{'='*100}")
                        break

        if use_strict_val:
            # save a temp ckpt for eval
            tmp_ckpt = os.path.join(args.save_dir, "_student_tmp_for_val.pth")
            torch.save(student.state_dict(), tmp_ckpt)
            try:
                metrics = _eval_val_by_eval_retrieval(base_cfg_path, tmp_ckpt, args, device=str(device))
            except Exception as e:
                print(f"[val][strict] failed: {e}", flush=True)
                metrics = None
            if metrics:
                score = float(metrics.get(args.monitor, float("-inf")))
                print(f"[val][strict] epoch={ep} recall@1={metrics['recall@1']:.4f} map@10={metrics['map@10']:.4f} monitor={args.monitor} score={score:.4f}", flush=True)

                improved = score > best_score + 1e-8
                if improved:
                    best_score = score
                    bad_epochs = 0
                    if args.save_best:
                        best_path = os.path.join(args.save_dir, "student_best.pth")
                        torch.save(student.state_dict(), best_path)
                        print(f"[OK] saved best -> {best_path} (best_score={best_score:.4f})", flush=True)

                        # additionally write best_metrics.json
                        import json

                        with open(os.path.join(args.save_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump({"epoch": ep, "monitor": args.monitor, "score": best_score, **metrics}, f, ensure_ascii=False, indent=2)
                else:
                    bad_epochs += 1
                    if args.patience and bad_epochs >= int(args.patience):
                        print(f"[EARLY-STOP] no improvement for {bad_epochs} epochs (patience={args.patience}).", flush=True)
                        break

            # cleanup tmp ckpt
            try:
                if os.path.exists(tmp_ckpt):
                    os.remove(tmp_ckpt)
            except Exception:
                pass

    # Ensure we always have a final checkpoint file
    # If we have best, create a symlink/copy to student_ep{total_epochs}.pth for run_pipeline.py
    final_ep_path = os.path.join(args.save_dir, f"student_ep{args.epochs}.pth")

    print(f"\n{'='*100}")
    print("TRAINING COMPLETE")
    print(f"{'='*100}")

    if best_path and os.path.exists(best_path):
        print(f"✓ Best model: {best_path}")
        print(f"  Best score ({args.monitor}): {best_score:.4f}")
        # Also save as the expected final epoch checkpoint
        if not os.path.exists(final_ep_path):
            import shutil
            shutil.copy(best_path, final_ep_path)
            print(f"  Copied to: {final_ep_path}")
    elif os.path.exists(ckpt_path):
        print("done. last student checkpoint:", ckpt_path)
        # Copy to expected final epoch if needed
        if ckpt_path != final_ep_path and not os.path.exists(final_ep_path):
            import shutil
            shutil.copy(ckpt_path, final_ep_path)
            print(f"[INFO] copied last checkpoint to: {final_ep_path}")
    else:
        # Fallback: find the latest checkpoint
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.save_dir, "student_ep*.pth")))
        if ckpts:
            latest = ckpts[-1]
            print(f"done. latest checkpoint: {latest}")
            if not os.path.exists(final_ep_path):
                import shutil
                shutil.copy(latest, final_ep_path)
                print(f"[INFO] copied latest checkpoint to: {final_ep_path}")
        else:
            print("[WARNING] No checkpoint found!")


if __name__ == "__main__":
    main()

