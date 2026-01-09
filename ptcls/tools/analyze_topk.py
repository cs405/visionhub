"""TopK 分析工具 - 分析检索评估的 topK 结果，找出问题类别

使用方式:
python visionhub/tools/analyze_topk.py \
  --topk_json visionhub/output/eval/my_exp/topk.json \
  --data_yaml dataset/data.yaml \
  --save_dir visionhub/output/eval/my_exp/analysis

功能：
1. 统计每个类别的检索成功率
2. 找出最难检索的类别（低召回率）
3. 找出最容易混淆的类别对
4. 可视化混淆矩阵
5. 生成详细的分析报告
"""

import argparse
import json
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple


def load_class_names(data_yaml: str) -> Dict[int, str]:
    """从 YOLO data.yaml 加载类别名称映射"""
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names


def analyze_per_class_recall(topk_data: List[dict], class_names: Dict[int, str], k: int = 1) -> Dict:
    """分析每个类别的 Recall@K"""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for item in topk_data:
        query_label = item.get('query_cls_id', item.get('label'))  # Support both formats
        topk_labels = [r.get('gallery_cls_id', r.get('label')) for r in item['topk']][:k]

        class_total[query_label] += 1
        if query_label in topk_labels:
            class_correct[query_label] += 1

    results = []
    for label in sorted(class_total.keys()):
        total = class_total[label]
        correct = class_correct[label]
        recall = correct / total if total > 0 else 0.0

        results.append({
            'class_id': label,
            'class_name': class_names.get(label, f'class_{label}'),
            'total': total,
            'correct': correct,
            'recall@{}'.format(k): recall
        })

    return results


def find_confused_pairs(topk_data: List[dict], class_names: Dict[int, str], k: int = 5) -> List[Tuple]:
    """找出最容易混淆的类别对"""
    confusion_matrix = defaultdict(int)

    for item in topk_data:
        query_label = item.get('query_cls_id', item.get('label'))
        topk_labels = [r.get('gallery_cls_id', r.get('label')) for r in item['topk']][:k]

        # 如果 top1 不是正确类别，记录混淆
        if topk_labels and topk_labels[0] != query_label:
            confusion_matrix[(query_label, topk_labels[0])] += 1

    # 排序找出最容易混淆的 pair
    confused_pairs = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)

    results = []
    for (true_label, pred_label), count in confused_pairs[:20]:  # 前 20 对
        results.append({
            'true_class_id': true_label,
            'true_class_name': class_names.get(true_label, f'class_{true_label}'),
            'confused_with_id': pred_label,
            'confused_with_name': class_names.get(pred_label, f'class_{pred_label}'),
            'count': count
        })

    return results


def analyze_difficult_queries(topk_data: List[dict], class_names: Dict[int, str]) -> List[dict]:
    """找出最难检索的 query（top5 都没有正确答案）"""
    difficult = []

    for item in topk_data:
        query_label = item.get('query_cls_id', item.get('label'))
        topk_labels = [r.get('gallery_cls_id', r.get('label')) for r in item['topk']][:5]

        if query_label not in topk_labels:
            query_name = item.get('query_name', item.get('query_image_id', 'unknown'))
            difficult.append({
                'query_name': query_name,
                'true_class_id': query_label,
                'true_class_name': class_names.get(query_label, f'class_{query_label}'),
                'top5_classes': [
                    {
                        'id': r.get('gallery_cls_id', r.get('label')),
                        'name': class_names.get(r.get('gallery_cls_id', r.get('label')), f"class_{r.get('gallery_cls_id', r.get('label'))}"),
                        'score': r.get('score', 0.0)
                    }
                    for r in item['topk'][:5]
                ]
            })

    return difficult


def generate_report(
    per_class_recall: List[dict],
    confused_pairs: List[dict],
    difficult_queries: List[dict],
    save_dir: Path
):
    """生成分析报告"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细 JSON
    with open(save_dir / 'per_class_recall.json', 'w', encoding='utf-8') as f:
        json.dump(per_class_recall, f, indent=2, ensure_ascii=False)

    with open(save_dir / 'confused_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(confused_pairs, f, indent=2, ensure_ascii=False)

    with open(save_dir / 'difficult_queries.json', 'w', encoding='utf-8') as f:
        json.dump(difficult_queries, f, indent=2, ensure_ascii=False)

    # 生成文本报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("检索评估 TopK 分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. 每类别召回率（按召回率排序，找出最差的）
    report_lines.append("## 1. 每类别 Recall@1 分析")
    report_lines.append("-" * 80)
    sorted_recall = sorted(per_class_recall, key=lambda x: x['recall@1'])

    report_lines.append("\n### 召回率最低的 10 个类别：")
    for item in sorted_recall[:10]:
        report_lines.append(
            f"  {item['class_name']:30s} (id={item['class_id']:3d}): "
            f"Recall@1={item['recall@1']:.4f} ({item['correct']}/{item['total']})"
        )

    report_lines.append("\n### 召回率最高的 10 个类别：")
    for item in sorted_recall[-10:][::-1]:
        report_lines.append(
            f"  {item['class_name']:30s} (id={item['class_id']:3d}): "
            f"Recall@1={item['recall@1']:.4f} ({item['correct']}/{item['total']})"
        )

    # 2. 最容易混淆的类别对
    report_lines.append("\n" + "=" * 80)
    report_lines.append("## 2. 最容易混淆的类别对（Top1 预测错误统计）")
    report_lines.append("-" * 80)
    for i, item in enumerate(confused_pairs[:15], 1):
        report_lines.append(
            f"{i:2d}. {item['true_class_name']:20s} -> {item['confused_with_name']:20s}  "
            f"(混淆次数: {item['count']})"
        )

    # 3. 最难检索的 query
    report_lines.append("\n" + "=" * 80)
    report_lines.append("## 3. 最难检索的 Query（Top5 都没有正确答案）")
    report_lines.append("-" * 80)
    report_lines.append(f"总计: {len(difficult_queries)} 个 query")

    if difficult_queries:
        report_lines.append("\n前 10 个示例：")
        for i, item in enumerate(difficult_queries[:10], 1):
            report_lines.append(f"\n{i}. {item['query_name']}")
            report_lines.append(f"   真实类别: {item['true_class_name']} (id={item['true_class_id']})")
            report_lines.append(f"   Top5 预测:")
            for j, pred in enumerate(item['top5_classes'], 1):
                report_lines.append(f"     {j}. {pred['name']:20s} (id={pred['id']:3d}, score={pred['score']:.4f})")

    # 4. 统计总结
    report_lines.append("\n" + "=" * 80)
    report_lines.append("## 4. 统计总结")
    report_lines.append("-" * 80)

    avg_recall = np.mean([x['recall@1'] for x in per_class_recall])
    median_recall = np.median([x['recall@1'] for x in per_class_recall])

    report_lines.append(f"平均 Recall@1: {avg_recall:.4f}")
    report_lines.append(f"中位数 Recall@1: {median_recall:.4f}")
    report_lines.append(f"总类别数: {len(per_class_recall)}")
    report_lines.append(f"Recall@1 = 0 的类别数: {sum(1 for x in per_class_recall if x['recall@1'] == 0)}")
    report_lines.append(f"Recall@1 = 1 的类别数: {sum(1 for x in per_class_recall if x['recall@1'] == 1.0)}")

    # 保存文本报告
    report_text = "\n".join(report_lines)
    with open(save_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n[OK] 分析报告已保存到: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="分析检索评估的 topK 结果")
    parser.add_argument('--topk_json', type=str, required=True, help='topk.json 文件路径')
    parser.add_argument('--data_yaml', type=str, required=True, help='YOLO data.yaml 文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='分析结果保存目录')
    args = parser.parse_args()

    # 加载数据
    print(f"[INFO] 加载 topk 结果: {args.topk_json}")
    with open(args.topk_json, 'r', encoding='utf-8') as f:
        topk_data = json.load(f)

    print(f"[INFO] 加载类别名称: {args.data_yaml}")
    class_names = load_class_names(args.data_yaml)

    print(f"[INFO] 开始分析...")

    # 分析
    per_class_recall = analyze_per_class_recall(topk_data, class_names, k=1)
    confused_pairs = find_confused_pairs(topk_data, class_names, k=5)
    difficult_queries = analyze_difficult_queries(topk_data, class_names)

    # 生成报告
    save_dir = Path(args.save_dir)
    generate_report(per_class_recall, confused_pairs, difficult_queries, save_dir)


if __name__ == '__main__':
    main()

