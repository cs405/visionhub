"""tools/eval_retrieval.py

严格检索评估脚本（对齐 visionhub Shitu 的评估思路，Professional）。

支持：
- train/val 分离：你可以指定 val 目录(或 yolo val images/labels)
- query/gallery 分离：query 集与 gallery 集完全分离
- 读取 YOLO data.yaml names 映射（class_id <-> class_name）
- 输出完整指标：Recall@{1,5,10}、mAP@{10,all}

两种输入格式（任意一种即可）：
1) folder cls 格式：
   query_root/
     class_a/*.jpg
   gallery_root/
     class_a/*.jpg

2) YOLO det 格式（会按 bbox crop 成 patch 再提 embedding）：
   query_images + query_labels
   gallery_images + gallery_labels
   data_yaml（可选，但建议提供，用于打印类名）

用法示例：
python visionhub/tools/eval_retrieval.py -c visionhub/configs/shitu/rec_faiss_demo.yaml \
  --gallery_images dataset/images/val --gallery_labels dataset/labels/val \
  --query_images dataset/images/test --query_labels dataset/labels/test \
  --data_yaml dataset/data.yaml

注意：
- 模型权重从配置 Global.rec_inference_model_dir 加载。
- 评估只跑 rec embedding，不跑 YOLO 检测。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.data.datasets.image_folder_pairs import ImageFolderDataset
from ptcls.data.datasets.yolo_det_crop_dataset import YoloDetCropDataset
from ptcls.metric.retrieval_eval import map_at_k, recall_at_k
from ptcls.rec.predictor import RecPredictor
from ptcls.utils import config as cfg_mod
from ptcls.utils.yolo_data import load_yolo_data_yaml
from ptcls.utils.io import ensure_dir, save_json

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)

    # folder cls
    p.add_argument("--gallery_root", default=None)
    p.add_argument("--query_root", default=None)

    # yolo det
    p.add_argument("--gallery_images", default=None)
    p.add_argument("--gallery_labels", default=None)
    p.add_argument("--query_images", default=None)
    p.add_argument("--query_labels", default=None)
    p.add_argument("--data_yaml", default=None)

    p.add_argument("--max_gallery", type=int, default=5000)
    p.add_argument("--max_query", type=int, default=2000)
    p.add_argument("--map_k", type=int, default=10)

    # strict eval options
    p.add_argument("--strict_image_split", action="store_true", help="use image-level split (one query per image)")
    p.add_argument(
        "--strict_pick",
        choices=["first", "random", "max"],
        default="first",
        help="when strict_image_split, which bbox to pick per image",
    )
    p.add_argument("--class_ids", default=None, help="optional: only evaluate these class ids, like '0,2,5'")
    p.add_argument("--exclude_same_image", action="store_true", help="exclude gallery items from the same image as query")

    # output analysis
    p.add_argument("--save_dir", default=None, help="if set, save topk JSON and query visualization")
    p.add_argument("--topk", type=int, default=5, help="topK to export for each query")

    return p.parse_args()


def _extract_from_folder(ds: ImageFolderDataset, rec: RecPredictor, max_n: int) -> Tuple[np.ndarray, List[int]]:
    feats = []
    labels = []
    n = min(len(ds), max_n)
    for i in range(n):
        img, y = ds[i]
        img_np = np.array(img)  # RGB
        feat = rec.predict(img_np, feature_normalize=True)
        feats.append(feat[0])
        labels.append(int(y))
    return np.stack(feats, axis=0), labels


def _extract_from_yolo(
    ds: YoloDetCropDataset,
    rec: RecPredictor,
    max_n: int,
    strict_image_split: bool,
    strict_pick: str,
    class_ids: List[int] | None,
):
    """返回:
    - feats: [N,D]
    - labels: list[int]
    - image_ids: list[str]
    - metas: list[dict]  (用于导出 topk/可视化)
    """

    feats = []
    labels = []
    image_ids = []
    metas = []

    if not strict_image_split:
        n = min(len(ds), max_n)
        for i in range(n):
            item = ds[i]
            img = item[0]
            y = int(item[1])
            image_id = str(item[-2])
            bbox = item[-1]
            if class_ids is not None and y not in class_ids:
                continue
            img_np = np.array(img)
            feat = rec.predict(img_np, feature_normalize=True)
            feats.append(feat[0])
            labels.append(y)
            image_ids.append(image_id)
            metas.append({"image_id": image_id, "bbox": list(map(int, bbox)), "cls_id": y})

        return np.stack(feats, axis=0), labels, image_ids, metas

    # strict: one object per image
    import random

    by_img = {}
    for s in ds.samples:
        if class_ids is not None and int(s.cls_id) not in class_ids:
            continue
        by_img.setdefault(s.image_id, []).append(s)

    img_ids = sorted(by_img.keys())[: max_n]

    from PIL import Image as _PILImage

    for image_id in img_ids:
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
        image_ids.append(str(image_id))
        metas.append({"image_id": str(image_id), "bbox": [int(x1), int(y1), int(x2), int(y2)], "cls_id": int(s.cls_id)})

    return np.stack(feats, axis=0), labels, image_ids, metas


def main():
    args = parse_args()
    cfg = cfg_mod.get_config(args.config, overrides=None, show=False)

    rec = RecPredictor(cfg)

    class_ids = None
    if args.class_ids:
        class_ids = [int(x) for x in str(args.class_ids).split(",") if str(x).strip() != ""]

    id2name = None
    if args.data_yaml:
        id2name, _ = load_yolo_data_yaml(args.data_yaml)

    use_folder = args.gallery_root and args.query_root
    use_yolo = args.gallery_images and args.gallery_labels and args.query_images and args.query_labels

    if not (use_folder or use_yolo):
        raise ValueError(
            "Provide either --gallery_root/--query_root OR YOLO det args (--gallery_images/--gallery_labels/--query_images/--query_labels)"
        )

    if use_folder:
        g_ds = ImageFolderDataset(args.gallery_root)
        q_ds = ImageFolderDataset(args.query_root)
        g_feat, g_lab = _extract_from_folder(g_ds, rec, args.max_gallery)
        q_feat, q_lab = _extract_from_folder(q_ds, rec, args.max_query)
        g_img_ids, q_img_ids, g_metas, q_metas = None, None, None, None
    else:
        g_ds = YoloDetCropDataset(args.gallery_images, args.gallery_labels, data_yaml=args.data_yaml, return_name=False)
        q_ds = YoloDetCropDataset(args.query_images, args.query_labels, data_yaml=args.data_yaml, return_name=False)

        g_feat, g_lab, g_img_ids, g_metas = _extract_from_yolo(
            g_ds,
            rec,
            args.max_gallery,
            strict_image_split=args.strict_image_split,
            strict_pick=args.strict_pick,
            class_ids=class_ids,
        )
        q_feat, q_lab, q_img_ids, q_metas = _extract_from_yolo(
            q_ds,
            rec,
            args.max_query,
            strict_image_split=args.strict_image_split,
            strict_pick=args.strict_pick,
            class_ids=class_ids,
        )

    # similarity
    g_feat = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-12)
    q_feat = q_feat / (np.linalg.norm(q_feat, axis=1, keepdims=True) + 1e-12)
    sim = q_feat @ g_feat.T

    if use_yolo and args.exclude_same_image:
        q_ids = np.asarray(q_img_ids, dtype=object)
        g_ids = np.asarray(g_img_ids, dtype=object)
        same = q_ids[:, None] == g_ids[None, :]
        sim = sim.copy()
        sim[same] = -1e9

    # metrics
    r = recall_at_k(sim, q_lab, g_lab, ks=(1, 5, 10))
    m10 = map_at_k(sim, q_lab, g_lab, k=args.map_k)
    mall = map_at_k(sim, q_lab, g_lab, k=None)

    print("=== Retrieval Eval ===")
    print(f"#gallery={len(g_lab)}  #query={len(q_lab)}")
    print(f"Recall@1={r[1]:.4f}  Recall@5={r[5]:.4f}  Recall@10={r[10]:.4f}")
    print(f"mAP@{args.map_k}={m10:.4f}  mAP@all={mall:.4f}")

    # persist metrics
    if args.save_dir:
        out_dir = ensure_dir(args.save_dir)
        metrics = {
            "gallery_size": int(len(g_lab)),
            "query_size": int(len(q_lab)),
            "recall@1": float(r[1]),
            "recall@5": float(r[5]),
            "recall@10": float(r[10]),
            f"mAP@{int(args.map_k)}": float(m10),
            "mAP@all": float(mall),
            "strict_image_split": bool(args.strict_image_split),
            "strict_pick": str(args.strict_pick),
            "exclude_same_image": bool(args.exclude_same_image),
            "class_ids": args.class_ids,
        }
        save_json(os.path.join(out_dir, "metrics.json"), metrics)

    # export topk
    if use_yolo and args.save_dir:
        out_dir = ensure_dir(args.save_dir)
        order = np.argsort(-sim, axis=1)

        exports = []
        for qi in range(sim.shape[0]):
            q_meta = q_metas[qi]
            q_cls_id = int(q_meta["cls_id"])
            q_cls_name = id2name.get(q_cls_id, str(q_cls_id)) if id2name else str(q_cls_id)

            topk_idx = order[qi, : args.topk].tolist()
            matches = []
            for rank, gi in enumerate(topk_idx, start=1):
                g_meta = g_metas[gi]
                g_cls_id = int(g_meta["cls_id"])
                g_cls_name = id2name.get(g_cls_id, str(g_cls_id)) if id2name else str(g_cls_id)
                matches.append(
                    {
                        "rank": rank,
                        "score": float(sim[qi, gi]),
                        "gallery_image_id": g_meta["image_id"],
                        "gallery_bbox": g_meta["bbox"],
                        "gallery_cls_id": g_cls_id,
                        "gallery_cls_name": g_cls_name,
                    }
                )

            exports.append(
                {
                    "query_idx": qi,
                    "query_image_id": q_meta["image_id"],
                    "query_bbox": q_meta["bbox"],
                    "query_cls_id": q_cls_id,
                    "query_cls_name": q_cls_name,
                    "topk": matches,
                }
            )

        save_json(os.path.join(out_dir, "topk.json"), exports)
        print(f"[OK] saved topk.json -> {os.path.join(out_dir, 'topk.json')}")

    # --- Per-class Recall@1 (query-side) ---
    if id2name is not None:
        q_lab_np = np.asarray(q_lab)
        order = np.argsort(-sim, axis=1)
        top1 = order[:, 0]
        g_lab_np = np.asarray(g_lab)
        hit = (g_lab_np[top1] == q_lab_np)
        cls_ids = np.unique(q_lab_np)
        print("--- Per-class Recall@1 (query-side) ---")
        for cid in cls_ids:
            m = q_lab_np == cid
            r1 = float(hit[m].mean()) if m.any() else 0.0
            print(f"{cid:>3} {id2name.get(int(cid), str(cid)):<20} r@1={r1:.4f} (n={int(m.sum())})")


if __name__ == "__main__":
    main()

