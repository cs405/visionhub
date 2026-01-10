# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from ...utils.avgmeter import AverageMeter


def face_recognition_eval(engine, epoch_id=0):
    """人脸识别模型评估

    评估人脸识别模型在人脸验证任务上的性能。

    Args:
        engine: 评估引擎
        epoch_id: epoch ID

    Returns:
        main_metric: 主要指标值
    """
    # 重置指标
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()

    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter("batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,"),
    }

    print_batch_step = engine.config.get("Global", {}).get("print_batch_step", 10)

    tic = time.time()
    accum_samples = 0
    total_samples = len(engine.eval_dataloader.dataset)
    max_iter = len(engine.eval_dataloader)

    # 获取配置
    flip_test = engine.config.get("Global", {}).get("flip_test", False)
    feature_normalize = engine.config.get("Global", {}).get("feature_normalize", False)

    # 设置为评估模式
    engine.model.eval()

    with torch.no_grad():
        for iter_id, batch in enumerate(engine.eval_dataloader):
            if iter_id >= max_iter:
                break

            if iter_id == 5:
                for key in time_info:
                    time_info[key].reset()

            time_info["reader_cost"].update(time.time() - tic)

            # 解包批次数据
            images_left, images_right, labels = [
                x.to(engine.device) if isinstance(x, torch.Tensor)
                else torch.tensor(x).to(engine.device)
                for x in batch[:3]
            ]
            batch_remains = [
                x.to(engine.device) if isinstance(x, torch.Tensor)
                else torch.tensor(x).to(engine.device)
                for x in batch[3:]
            ]
            labels = labels.long()
            batch_size = images_left.shape[0]

            # 翻转图像（如果启用）
            if flip_test:
                images_left = torch.cat(
                    [images_left, torch.flip(images_left, dims=[-1])], dim=0
                )
                images_right = torch.cat(
                    [images_right, torch.flip(images_right, dims=[-1])], dim=0
                )

            # 前向传播
            out_left = engine.model(images_left)
            out_right = engine.model(images_right)

            # 获取特征
            retrieval_feature_from = engine.config.get("Global", {}).get(
                "retrieval_feature_from", "features"
            )
            if retrieval_feature_from == "features":
                # 使用 neck 输出作为特征
                embeddings_left = (out_left["features"] if isinstance(out_left, dict)
                                  else out_left)
                embeddings_right = (out_right["features"] if isinstance(out_right, dict)
                                   else out_right)
            else:
                # 使用 backbone 输出作为特征
                embeddings_left = (out_left["backbone"] if isinstance(out_left, dict)
                                  else out_left)
                embeddings_right = (out_right["backbone"] if isinstance(out_right, dict)
                                   else out_right)

            # 特征归一化
            if feature_normalize:
                embeddings_left = F.normalize(embeddings_left, p=2, dim=1)
                embeddings_right = F.normalize(embeddings_right, p=2, dim=1)

            # 融合翻转图像的特征
            if flip_test:
                embeddings_left = (embeddings_left[:batch_size] +
                                  embeddings_left[batch_size:])
                embeddings_right = (embeddings_right[:batch_size] +
                                   embeddings_right[batch_size:])

            # 分布式采样问题处理
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            current_samples = batch_size * world_size
            accum_samples += current_samples

            # 分布式收集
            if dist.is_initialized() and world_size > 1:
                # 收集 embeddings
                emb_left_list = [torch.zeros_like(embeddings_left) for _ in range(world_size)]
                dist.all_gather(emb_left_list, embeddings_left)
                embeddings_left = torch.cat(emb_left_list, dim=0)

                emb_right_list = [torch.zeros_like(embeddings_right) for _ in range(world_size)]
                dist.all_gather(emb_right_list, embeddings_right)
                embeddings_right = torch.cat(emb_right_list, dim=0)

                # 收集 labels
                label_list = [torch.zeros_like(labels) for _ in range(world_size)]
                dist.all_gather(label_list, labels)
                labels = torch.cat(label_list, dim=0)

                # 收集其他数据
                batch_remains_gathered = []
                for x in batch_remains:
                    x_list = [torch.zeros_like(x) for _ in range(world_size)]
                    dist.all_gather(x_list, x)
                    batch_remains_gathered.append(torch.cat(x_list, dim=0))
                batch_remains = batch_remains_gathered

                # 处理重复采样
                if accum_samples > total_samples:
                    rest_num = total_samples + current_samples - accum_samples
                    embeddings_left = embeddings_left[:rest_num]
                    embeddings_right = embeddings_right[:rest_num]
                    labels = labels[:rest_num]
                    batch_remains = [x[:rest_num] for x in batch_remains]

            # 计算指标
            if engine.eval_metric_func is not None:
                engine.eval_metric_func(
                    embeddings_left, embeddings_right, labels, *batch_remains
                )

            time_info["batch_cost"].update(time.time() - tic)

            # 打印日志
            if iter_id % print_batch_step == 0:
                time_msg = "s, ".join([
                    f"{key}: {time_info[key].avg:.5f}"
                    for key in time_info
                ])

                ips_msg = f"ips: {batch_size / time_info['batch_cost'].avg:.5f} images/sec"

                metric_msg = ", ".join([
                    f"{key}: {output_info[key].val:.5f}"
                    for key in output_info
                ])

                log_msg = (f"[Eval][Epoch {epoch_id}][Iter: {iter_id}/{len(engine.eval_dataloader)}]"
                          f"{metric_msg}, {time_msg}, {ips_msg}")

                if hasattr(engine, 'logger'):
                    engine.logger.info(log_msg)
                else:
                    print(log_msg)

            tic = time.time()

    # 获取最终指标
    metric_msg = ", ".join([
        f"{key}: {output_info[key].avg:.5f}"
        for key in output_info
    ])

    if hasattr(engine.eval_metric_func, 'avg_info'):
        metric_msg += f", {engine.eval_metric_func.avg_info}"

    if hasattr(engine, 'logger'):
        engine.logger.info(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")
    else:
        print(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")

    # 恢复训练模式
    engine.model.train()

    # 返回第一个指标
    if engine.eval_metric_func is None:
        return -1

    # 尝试从指标函数获取主指标
    if hasattr(engine.eval_metric_func, 'avg'):
        return engine.eval_metric_func.avg()
    else:
        return 0.0

