"""run_pipeline.py

Windows-friendly one-click pipeline:
- KD train (student embedding)
- patch config (Global.rec_inference_model_dir)
- rebuild faiss index
- retrieval eval

This is a thin, stable wrapper around existing scripts in `visionhub/tools`.

Why not just call `visionhub/tools/shitu_kd_reindex_eval.py`?
- This wrapper adds: retries, skip-existing, absolute-path normalization,
  optional cpu-safe defaults, and artifact checks.

Examples (PowerShell):
python run_pipeline.py all -c visionhub/configs/shitu/rec_faiss_demo.yaml `
  --data_yaml dataset/data.yaml `
  --yolo_train_images dataset/images/train --yolo_train_labels dataset/labels/train `
  --eval_gallery_images dataset/images/val --eval_gallery_labels dataset/labels/val `
  --eval_query_images dataset/images/test --eval_query_labels dataset/labels/test `
  --save_dir visionhub/output_rec_kd --epochs 1 --device cpu --num_workers 0 `
  --strict_image_split --exclude_same_image --save_eval_dir visionhub/output_rec_kd/eval

"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Artifacts:
    ckpt: Path
    index_dir: Path
    vector_index: Path
    id_map: Path


def _abspath(p: str | None) -> str | None:
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())


def _maybe_fix_yolo_split_path(p: str) -> str:
    """Try to map common YOLO paths to this repo's dataset layout.

    Your dataset layout is often:
      dataset/train/images, dataset/train/labels
    But users may pass:
      dataset/images/train, dataset/labels/train

    This helper only changes the path if the original doesn't exist.
    """
    pp = Path(p).expanduser().resolve()
    if pp.exists():
        return str(pp)

    # attempt: dataset/images/train  -> dataset/train
    parts = [x.lower() for x in pp.parts]
    try:
        idx = parts.index("dataset")
    except ValueError:
        return str(pp)

    # If endswith dataset\images\train, map to dataset\train
    if len(parts) >= idx + 3:
        tail3 = parts[-3:]
        if tail3[-3] in ("images", "labels") and tail3[-2] in ("train", "val", "test"):
            # .../dataset/(images|labels)/(split)
            split = pp.parts[-1]
            cand = Path(*pp.parts[:-2]) / split
            if cand.exists() and cand.is_dir():
                return str(cand.resolve())
    return str(pp)


def _is_truthy_file(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _run(cmd: list[str], *, stage: str, retries: int, sleep_sec: float) -> None:
    last_rc = None
    for attempt in range(retries + 1):
        print(f"\n[STAGE:{stage}] attempt {attempt + 1}/{retries + 1}")
        print("[RUN]", " ".join(cmd), flush=True)
        print()  # Empty line for readability

        # Real-time output: inherit stdout/stderr so user can see progress
        p = subprocess.run(cmd, stdout=None, stderr=None)

        last_rc = p.returncode
        if p.returncode == 0:
            return

        # On failure, print error
        print(f"\n[ERROR] stage={stage} failed rc={p.returncode}", flush=True)

        if attempt < retries:
            wait = sleep_sec * (2**attempt)
            print(f"[WARN] stage={stage} failed rc={p.returncode}, retry in {wait:.1f}s", flush=True)
            time.sleep(wait)

    raise SystemExit(last_rc if last_rc is not None else 1)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _visionhub_root() -> Path:
    return _project_root() / "ptcls"


def _tool(script: str) -> Path:
    return _visionhub_root() / script


def _default_index_paths(cfg_path: Path) -> tuple[Path, Path, Path]:
    # Best-effort parse of IndexProcess.index_dir; fall back to default in config.
    # We avoid adding dependencies; this wrapper is about orchestration.
    index_dir = _project_root() / "label_gallery" / "index"
    vector_index = index_dir / "vector.index"
    id_map = index_dir / "id_map.pkl"
    return index_dir, vector_index, id_map


def _expected_ckpt(save_dir: Path, epochs: int) -> Path:
    # Prefer best checkpoint if present
    best = save_dir / "student_best.pth"
    if best.exists() and best.is_file() and best.stat().st_size > 0:
        return best
    return save_dir / f"student_ep{epochs}.pth"


def _resolve_final_ckpt(save_dir: Path, epochs: int) -> Path:
    """Resolve which checkpoint should be used after training.

    Handles early-stop case:
    - if student_best.pth exists, use it
    - else use the latest student_ep*.pth if present
    - else fall back to expected student_ep{epochs}.pth
    """
    best = save_dir / "student_best.pth"
    if _is_truthy_file(best):
        return best

    eps = sorted(save_dir.glob("student_ep*.pth"), key=lambda p: p.stat().st_mtime)
    for p in reversed(eps):
        if _is_truthy_file(p):
            return p

    return _expected_ckpt(save_dir, epochs)


def _cmd_train(args) -> list[str]:
    py = sys.executable
    cmd = [
        py,
        str(_tool("train_rec_kd.py")),
        "--save_dir",
        _abspath(args.save_dir),
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
        _maybe_fix_yolo_split_path(_abspath(args.yolo_train_images)),
        "--yolo_labels",
        _maybe_fix_yolo_split_path(_abspath(args.yolo_train_labels)),
        "--eval_every",
        "0",
    ]

    if args.data_yaml:
        cmd += ["--data_yaml", _abspath(args.data_yaml)]
    if args.amp:
        cmd += ["--amp"]
    if args.use_pk:
        cmd += ["--use_pk", "--P", str(args.P), "--K", str(args.K)]

    cmd += [
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
    ]

    # extra metric losses
    if float(args.w_circle) > 0:
        cmd += [
            "--w_circle",
            str(args.w_circle),
            "--circle_m",
            str(args.circle_m),
            "--circle_gamma",
            str(args.circle_gamma),
        ]

    if float(args.w_arcface) > 0:
        cmd += [
            "--w_arcface",
            str(args.w_arcface),
            "--arcface_s",
            str(args.arcface_s),
            "--arcface_m",
            str(args.arcface_m),
        ]

    if args.teacher_pretrained:
        cmd += ["--teacher_pretrained"]
    if args.student_pretrained:
        cmd += ["--student_pretrained"]

    # val/early-stop
    if args.val_yolo_images and args.val_yolo_labels:
        cmd += [
            "--val_yolo_images",
            _maybe_fix_yolo_split_path(_abspath(args.val_yolo_images)),
            "--val_yolo_labels",
            _maybe_fix_yolo_split_path(_abspath(args.val_yolo_labels)),
            "--monitor",
            str(args.monitor),
        ]
        if args.save_best:
            cmd += ["--save_best"]
        if int(args.patience) > 0:
            cmd += ["--patience", str(args.patience)]

    if args.teacher_torchvision:
        cmd += ["--teacher_torchvision"]

    # strict val eval (train internal)
    if getattr(args, "val_gallery_images", None) and getattr(args, "val_gallery_labels", None) and getattr(args, "val_query_images", None) and getattr(args, "val_query_labels", None):
        cmd += [
            "--val_gallery_images",
            _maybe_fix_yolo_split_path(_abspath(args.val_gallery_images)),
            "--val_gallery_labels",
            _maybe_fix_yolo_split_path(_abspath(args.val_gallery_labels)),
            "--val_query_images",
            _maybe_fix_yolo_split_path(_abspath(args.val_query_images)),
            "--val_query_labels",
            _maybe_fix_yolo_split_path(_abspath(args.val_query_labels)),
        ]
        if getattr(args, "val_strict_image_split", False):
            cmd += ["--val_strict_image_split"]
        if getattr(args, "val_strict_pick", None):
            cmd += ["--val_strict_pick", str(args.val_strict_pick)]
        if getattr(args, "val_exclude_same_image", False):
            cmd += ["--val_exclude_same_image"]
        if getattr(args, "val_map_k", None) is not None:
            cmd += ["--val_map_k", str(args.val_map_k)]

    # training stability options
    if getattr(args, "skip_non_finite", False):
        cmd += ["--skip_non_finite"]
    if getattr(args, "grad_clip", 0.0) and float(args.grad_clip) > 0:
        cmd += ["--grad_clip", str(args.grad_clip)]
    if getattr(args, "max_skip_ratio", None) is not None:
        cmd += ["--max_skip_ratio", str(args.max_skip_ratio)]
    if getattr(args, "fallback_no_amp", False):
        cmd += ["--fallback_no_amp"]

    # warmup options
    if getattr(args, "warmup_epochs", None) is not None:
        cmd += ["--warmup_epochs", str(args.warmup_epochs)]
    if getattr(args, "warmup_mode", None):
        cmd += ["--warmup_mode", str(args.warmup_mode)]
    if getattr(args, "warmup_disable_supcon", False):
        cmd += ["--warmup_disable_supcon"]
    if getattr(args, "warmup_circle_gamma", None) is not None:
        cmd += ["--warmup_circle_gamma", str(args.warmup_circle_gamma)]

    return cmd


def _cmd_reindex(cfg_path: Path) -> list[str]:
    py = sys.executable
    return [py, str(_tool("build_gallery.py")), "-c", str(cfg_path)]


def _cmd_eval(args, cfg_path: Path) -> list[str]:
    py = sys.executable
    cmd = [
        py,
        str(_tool("eval_retrieval.py")),
        "-c",
        str(cfg_path),
        "--gallery_images",
        _maybe_fix_yolo_split_path(_abspath(args.eval_gallery_images)),
        "--gallery_labels",
        _maybe_fix_yolo_split_path(_abspath(args.eval_gallery_labels)),
        "--query_images",
        _maybe_fix_yolo_split_path(_abspath(args.eval_query_images)),
        "--query_labels",
        _maybe_fix_yolo_split_path(_abspath(args.eval_query_labels)),
        "--topk",
        str(args.topk),
        "--map_k",
        str(args.map_k),
        "--max_gallery",
        str(args.max_gallery),
        "--max_query",
        str(args.max_query),
    ]
    if args.data_yaml:
        cmd += ["--data_yaml", _abspath(args.data_yaml)]
    if args.strict_image_split:
        cmd += ["--strict_image_split", "--strict_pick", args.strict_pick]
    if args.exclude_same_image:
        cmd += ["--exclude_same_image"]
    if args.save_eval_dir:
        cmd += ["--save_dir", _abspath(args.save_eval_dir)]
    if args.class_ids:
        cmd += ["--class_ids", args.class_ids]
    return cmd


def _patch_config_set_ckpt(cfg_path: Path, ckpt: Path) -> Path:
    # Use existing YAML dependency in visionhub toolchain? Prefer stdlib-free.
    # We'll reuse PyYAML if available; it's already used in visionhub/tools.
    import yaml  # type: ignore

    cfg_path = cfg_path.resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    cfg.setdefault("Global", {})
    cfg["Global"]["rec_inference_model_dir"] = str(ckpt)

    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    with backup.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[OK] patched config: Global.rec_inference_model_dir={ckpt}")
    print(f"[OK] backup: {backup}")
    return backup


def _ensure_artifacts(art: Artifacts) -> None:
    missing: list[str] = []
    if not _is_truthy_file(art.ckpt):
        missing.append(f"ckpt: {art.ckpt}")
    if not _is_truthy_file(art.vector_index):
        missing.append(f"faiss index: {art.vector_index}")
    if not _is_truthy_file(art.id_map):
        missing.append(f"id map: {art.id_map}")
    if missing:
        raise FileNotFoundError("Missing/empty artifacts: " + ", ".join(missing))


def _cmd_predict_system(cfg_path: Path, override: list[str] | None = None) -> list[str]:
    py = sys.executable
    cmd = [py, str(_tool("predict_system.py")), "-c", str(cfg_path)]
    if override:
        cmd += ["-o"] + override
    return cmd


def _make_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)

    parent.add_argument("-c", "--config", required=True, help="visionhub shitu config yaml")

    parent.add_argument("--data_yaml", default=None)

    # train data
    parent.add_argument("--yolo_train_images", required=True)
    parent.add_argument("--yolo_train_labels", required=True)

    # global execution
    parent.add_argument("--retries", type=int, default=0)
    parent.add_argument("--retry_sleep", type=float, default=2.0)
    parent.add_argument("--skip_existing", action="store_true", help="skip stage if artifacts exist")

    # train args
    parent.add_argument("--save_dir", required=True)
    parent.add_argument("--epochs", type=int, default=5)
    parent.add_argument("--batch_size", type=int, default=16)
    parent.add_argument("--lr", type=float, default=1e-3)
    parent.add_argument("--device", default="cpu", help="cpu/cuda")
    parent.add_argument("--amp", action="store_true")
    parent.add_argument("--num_workers", type=int, default=0, help="Windows: 0 is most stable")

    # KD/loss
    parent.add_argument("--w_kd_embed", type=float, default=1.0)
    parent.add_argument("--w_supcon", type=float, default=0.2)
    parent.add_argument("--w_kd_logits", type=float, default=0.0)
    parent.add_argument("--temperature", type=float, default=4.0)
    parent.add_argument("--use_pk", action="store_true")
    parent.add_argument("--P", type=int, default=8)
    parent.add_argument("--K", type=int, default=4)
    parent.add_argument("--w_triplet", type=float, default=1.0)
    parent.add_argument("--triplet_margin", type=float, default=0.2)

    # metric learning extras
    parent.add_argument("--w_circle", type=float, default=0.0)
    parent.add_argument("--circle_m", type=float, default=0.25)
    parent.add_argument("--circle_gamma", type=float, default=256.0)

    parent.add_argument("--w_arcface", type=float, default=0.0)
    parent.add_argument("--arcface_s", type=float, default=64.0)
    parent.add_argument("--arcface_m", type=float, default=0.5)

    parent.add_argument("--teacher_pretrained", action="store_true")
    parent.add_argument("--student_pretrained", action="store_true")

    # optional system predict visualization
    parent.add_argument("--run_predict_system", action="store_true", help="run predict_system.py after reindex")
    parent.add_argument("--predict_infer_img", default=None, help="override Global.infer_imgs")
    parent.add_argument("--predict_save_path", default=None, help="override Global.save_visual_path")

    # eval options (shared; used by eval/all)
    parent.add_argument("--strict_image_split", action="store_true")
    parent.add_argument("--strict_pick", choices=["first", "random", "max"], default="max")
    parent.add_argument("--exclude_same_image", action="store_true")
    parent.add_argument("--save_eval_dir", default=None)
    parent.add_argument("--topk", type=int, default=5)
    parent.add_argument("--map_k", type=int, default=10)
    parent.add_argument("--max_gallery", type=int, default=5000)
    parent.add_argument("--max_query", type=int, default=2000)
    parent.add_argument("--class_ids", default=None)

    # val/early-stop (train side)
    parent.add_argument("--val_yolo_images", default=None)
    parent.add_argument("--val_yolo_labels", default=None)
    parent.add_argument("--monitor", choices=["map@10", "recall@1"], default="map@10")
    parent.add_argument("--save_best", action="store_true")
    parent.add_argument("--patience", type=int, default=0)
    parent.add_argument("--teacher_torchvision", action="store_true")

    # strict-val (train side, aligned with eval_retrieval)
    parent.add_argument("--val_gallery_images", default=None)
    parent.add_argument("--val_gallery_labels", default=None)
    parent.add_argument("--val_query_images", default=None)
    parent.add_argument("--val_query_labels", default=None)
    parent.add_argument("--val_strict_image_split", action="store_true")
    parent.add_argument("--val_strict_pick", choices=["first", "random", "max"], default="max")
    parent.add_argument("--val_exclude_same_image", action="store_true")
    parent.add_argument("--val_map_k", type=int, default=10)

    # training stability
    parent.add_argument("--skip_non_finite", action="store_true")
    parent.add_argument("--grad_clip", type=float, default=0.0)
    parent.add_argument("--max_skip_ratio", type=float, default=0.5)
    parent.add_argument("--fallback_no_amp", action="store_true")

    # warmup strategy (train side)
    parent.add_argument("--warmup_epochs", type=int, default=3)
    parent.add_argument("--warmup_mode", choices=["linear", "cosine", "step"], default="linear")
    parent.add_argument("--warmup_disable_supcon", action="store_true")
    parent.add_argument("--warmup_circle_gamma", type=float, default=None)

    p = argparse.ArgumentParser(prog="run_pipeline.py")
    sub = p.add_subparsers(dest="cmd", required=False)

    # kd/reindex need only train args
    kd = sub.add_parser("kd", parents=[parent], add_help=True)
    kd.set_defaults(cmd="kd")

    reindex = sub.add_parser("reindex", parents=[parent], add_help=True)
    reindex.set_defaults(cmd="reindex")

    # eval/all require eval dataset args (ONLY dataset args here to avoid conflicts)
    eval_parent = argparse.ArgumentParser(add_help=False)
    eval_parent.add_argument("--eval_gallery_images", required=True)
    eval_parent.add_argument("--eval_gallery_labels", required=True)
    eval_parent.add_argument("--eval_query_images", required=True)
    eval_parent.add_argument("--eval_query_labels", required=True)

    ev = sub.add_parser("eval", parents=[parent, eval_parent], add_help=True)
    ev.set_defaults(cmd="eval")

    allp = sub.add_parser("all", parents=[parent, eval_parent], add_help=True)
    allp.set_defaults(cmd="all")

    # lightweight predict command (only needs config)
    pred = sub.add_parser("predict", add_help=True)
    pred.add_argument("-c", "--config", required=True, help="visionhub shitu config yaml")
    pred.add_argument("--predict_infer_img", default=None, help="override Global.infer_imgs")
    pred.add_argument("--predict_save_path", default=None, help="override Global.save_visual_path")
    pred.add_argument("--retries", type=int, default=0)
    pred.add_argument("--retry_sleep", type=float, default=2.0)
    pred.set_defaults(cmd="predict")

    p.set_defaults(cmd="all")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _make_parser().parse_args(list(argv) if argv is not None else None)

    # standalone predict
    if args.cmd == "predict":
        cfg_path = Path(args.config).expanduser().resolve()
        overrides = []
        if args.predict_infer_img:
            overrides.append(f"Global.infer_imgs={_abspath(args.predict_infer_img)}")
        if args.predict_save_path:
            overrides.append(f"Global.save_visual_path={_abspath(args.predict_save_path)}")
        _run(
            _cmd_predict_system(cfg_path, overrides if overrides else None),
            stage="predict_system",
            retries=int(args.retries),
            sleep_sec=float(args.retry_sleep),
        )
        print("\n[DONE] predict finished")
        return

    cfg_path = Path(args.config).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _expected_ckpt(save_dir, args.epochs)
    index_dir, vector_index, id_map = _default_index_paths(cfg_path)
    art = Artifacts(ckpt=ckpt, index_dir=index_dir, vector_index=vector_index, id_map=id_map)

    def have_all_artifacts() -> bool:
        return _is_truthy_file(ckpt) and _is_truthy_file(vector_index) and _is_truthy_file(id_map)

    # KD stage
    if args.cmd in ("kd", "all"):
        if args.skip_existing and _is_truthy_file(ckpt):
            print(f"[SKIP] KD: checkpoint exists: {ckpt}")
        else:
            _run(_cmd_train(args), stage="kd", retries=args.retries, sleep_sec=args.retry_sleep)

        # Re-resolve checkpoint after training (early-stop may occur)
        ckpt = _resolve_final_ckpt(save_dir, args.epochs)
        art.ckpt = ckpt

        if not _is_truthy_file(ckpt):
            # show helpful directory listing
            try:
                existing = ", ".join([p.name for p in sorted(save_dir.glob("student_*.pth"))])
            except Exception:
                existing = "(failed to list)"
            raise FileNotFoundError(f"KD completed but checkpoint missing: {ckpt}. Existing: {existing}")

        _patch_config_set_ckpt(cfg_path, ckpt)

    # Reindex stage
    if args.cmd in ("reindex", "all"):
        if args.skip_existing and _is_truthy_file(vector_index) and _is_truthy_file(id_map):
            print(f"[SKIP] REINDEX: index exists: {vector_index}")
        else:
            _run(_cmd_reindex(cfg_path), stage="reindex", retries=args.retries, sleep_sec=args.retry_sleep)

        # optional: run predict_system to produce visualization
        if args.run_predict_system and args.cmd == "all":
            overrides = []
            if args.predict_infer_img:
                overrides.append(f"Global.infer_imgs={_abspath(args.predict_infer_img)}")
            if args.predict_save_path:
                overrides.append(f"Global.save_visual_path={_abspath(args.predict_save_path)}")
            _run(_cmd_predict_system(cfg_path, overrides if overrides else None), stage="predict_system", retries=args.retries, sleep_sec=args.retry_sleep)

    # Eval stage
    if args.cmd in ("eval", "all"):
        _run(_cmd_eval(args, cfg_path), stage="eval", retries=args.retries, sleep_sec=args.retry_sleep)

    # Final artifacts check for full pipeline
    if args.cmd == "all":
        art.ckpt = ckpt
        _ensure_artifacts(art)

    print("\n[DONE] pipeline finished")
    print(f" - ckpt: {ckpt}")
    print(f" - index: {vector_index}")
    if args.save_eval_dir:
        print(f" - eval_dir: {Path(args.save_eval_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()

