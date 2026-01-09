"""cleanup_outputs.py - Clean up old output_* directories in visionhub/

这个脚本用于清理 visionhub/ 下的旧输出文件夹（output_* 格式），
保留新的统一输出目录 visionhub/output/

使用方式：
  # 查看哪些文件夹会被清理（不实际删除）
  python visionhub/tools/cleanup_outputs.py --dry-run

  # 清理所有旧的 output_* 文件夹
  python visionhub/tools/cleanup_outputs.py

  # 交互式清理（逐个确认）
  python visionhub/tools/cleanup_outputs.py --interactive
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Clean up old output_* directories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要删除的文件夹，不实际删除"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="交互式确认每个文件夹是否删除"
    )
    parser.add_argument(
        "--root",
        default="visionhub",
        help="visionhub 根目录路径（默认：visionhub）"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] 目录不存在: {root}")
        return

    # 查找所有 output_* 开头的文件夹（排除统一的 output/ 文件夹）
    output_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("output_")
    ]

    if not output_dirs:
        print(f"[INFO] 没有找到需要清理的 output_* 文件夹在 {root}")
        return

    print(f"[INFO] 找到 {len(output_dirs)} 个旧输出文件夹：\n")

    for d in output_dirs:
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  - {d.name:40s}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n[DRY-RUN] 以上文件夹将被删除（本次未实际删除）")
        print("若要实际删除，请运行: python visionhub/tools/cleanup_outputs.py")
        return

    print()

    if not args.interactive:
        confirm = input("确认删除以上所有文件夹？[y/N]: ").strip().lower()
        if confirm != "y":
            print("[CANCELLED] 已取消操作")
            return

    deleted_count = 0
    skipped_count = 0

    for d in output_dirs:
        if args.interactive:
            choice = input(f"删除 {d.name}? [y/N/q(quit)]: ").strip().lower()
            if choice == "q":
                print("[CANCELLED] 用户中止")
                break
            if choice != "y":
                print(f"  [SKIP] {d.name}")
                skipped_count += 1
                continue

        try:
            shutil.rmtree(d)
            print(f"  [DELETED] {d.name}")
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] 删除失败 {d.name}: {e}")

    print(f"\n[DONE] 删除了 {deleted_count} 个文件夹，跳过 {skipped_count} 个")

    # 确保 visionhub/output/ 存在
    output_main = root / "output"
    if not output_main.exists():
        output_main.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 创建统一输出目录: {output_main}")

        # 创建子目录结构
        (output_main / "training").mkdir(exist_ok=True)
        (output_main / "eval").mkdir(exist_ok=True)
        (output_main / "inference").mkdir(exist_ok=True)
        print("[INFO] 创建子目录: training/, eval/, inference/")

    print("\n推荐阅读: visionhub/OUTPUT_MANAGEMENT.md 了解新的输出目录规范")


if __name__ == "__main__":
    main()

