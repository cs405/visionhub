# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import os
import json
import torch
import torch.distributed as dist


def save_predict_result(save_path, result):
    """保存预测结果为 JSON 文件

    Args:
        save_path (str): 保存路径，自动添加 .json 后缀
        result: 预测结果（可 JSON 序列化的对象）
    """
    if os.path.splitext(save_path)[-1] == '':
        if save_path[-1] == "/":
            save_path = save_path[:-1]
        save_path = save_path + '.json'
    elif os.path.splitext(save_path)[-1] == '.json':
        save_path = save_path
    else:
        raise Exception(
            f"{save_path} is invalid input path, only files in json format are supported."
        )

    if os.path.exists(save_path):
        print(f"Warning: The file {save_path} will be overwritten.")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def update_train_results(
    config,
    prefix,
    metric_info,
    done_flag=False,
    last_num=5,
    ema=False
):
    """更新训练结果元数据

    Args:
        config (dict): 配置字典
        prefix (str): 模型保存前缀，如 'best_model' 或 'epoch_10'
        metric_info (dict): 指标信息，包含 'metric' 键
        done_flag (bool): 训练是否完成
        last_num (int): 保留最近 N 个模型
        ema (bool): 是否使用 EMA
    """
    # 分布式训练只在主进程保存
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    assert last_num >= 1
    train_results_path = os.path.join(
        config["Global"]["output_dir"],
        "train_result.json"
    )

    # PyTorch 模型文件后缀
    save_model_tag = ["pth", "optimizer.pth"]
    save_inference_tag = ["inference.pth", "inference_config.json"]

    if ema:
        save_model_tag.append("ema.pth")

    # 读取或创建训练结果
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = config["Global"].get("model_name", None)
        train_results["label_dict"] = config.get("PostProcess", {}).get(
            "class_id_map_file", ""
        )
        train_results["train_log"] = "train.log"
        train_results["tensorboard_log"] = "runs/"
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}

    train_results["done_flag"] = done_flag

    # 更新模型信息
    if prefix == "best_model":
        train_results["models"]["best"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, f"{prefix}.{tag}"
            )
        for tag in save_inference_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, "inference", tag
            )
    else:
        # 移动历史模型记录
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = (
                train_results["models"][f"last_{i}"].copy()
            )

        train_results["models"]["last_1"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"]["last_1"][tag] = os.path.join(
                prefix, f"{prefix}.{tag}"
            )
        for tag in save_inference_tag:
            train_results["models"]["last_1"][tag] = os.path.join(
                prefix, "inference", tag
            )

    # 保存训练结果
    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp, indent=2, ensure_ascii=False)

