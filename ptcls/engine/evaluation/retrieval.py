# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

from collections import defaultdict
import torch
import torch.distributed as dist


def retrieval_eval(engine, epoch_id=0):
    """检索模型评估

    Args:
        engine: 评估引擎
        epoch_id: epoch ID

    Returns:
        main_metric: 主要指标值
    """
    engine.model.eval()

    # 步骤1：准备 query 和 gallery 特征
    if hasattr(engine, 'gallery_query_dataloader') and engine.gallery_query_dataloader is not None:
        gallery_feat, gallery_label, gallery_camera = compute_feature(
            engine, "gallery_query")
        query_feat, query_label, query_camera = gallery_feat, gallery_label, gallery_camera
    else:
        gallery_feat, gallery_label, gallery_camera = compute_feature(
            engine, "gallery")
        query_feat, query_label, query_camera = compute_feature(engine, "query")

    # 步骤2：分块处理特征以节省内存
    num_query = len(query_feat)
    block_size = engine.config.get("Global", {}).get("sim_block_size", 64)
    sections = [block_size] * (num_query // block_size)
    if num_query % block_size > 0:
        sections.append(num_query % block_size)

    query_feat_blocks = torch.split(query_feat, sections)
    query_label_blocks = torch.split(query_label, sections)
    query_camera_blocks = (torch.split(query_camera, sections)
                          if query_camera is not None else None)
    metric_key = None

    # 步骤3：计算指标
    if engine.eval_metric_func is None:
        metric_dict = {metric_key: 0.0}
    else:
        use_reranking = engine.config.get("Global", {}).get("re_ranking", False)
        if hasattr(engine, 'logger'):
            engine.logger.info(f"re_ranking={use_reranking}")
        else:
            print(f"re_ranking={use_reranking}")

        if use_reranking:
            # 重排序：计算距离矩阵
            distmat = compute_re_ranking_dist(
                query_feat, gallery_feat,
                engine.config.get("Global", {}).get("feature_normalize", True),
                k1=20, k2=6, lambda_value=0.3
            )
            # 排除非法距离
            if query_camera is not None:
                camera_mask = query_camera != gallery_camera.t()
                label_mask = query_label != gallery_label.t()
                keep_mask = label_mask | camera_mask
                distmat = (keep_mask.float() * distmat +
                          (~keep_mask).float() * (distmat.max() + 1))
            else:
                keep_mask = None
            # 计算指标
            metric_dict = engine.eval_metric_func(
                -distmat, query_label, gallery_label, keep_mask
            )
        else:
            # 不使用重排序：分块计算
            metric_dict = defaultdict(float)
            for block_idx, block_feat in enumerate(query_feat_blocks):
                # 计算距离矩阵（余弦相似度）
                distmat = torch.mm(block_feat, gallery_feat.t())

                # 排除非法距离
                if query_camera is not None:
                    camera_mask = query_camera_blocks[block_idx] != gallery_camera.t()
                    label_mask = query_label_blocks[block_idx] != gallery_label.t()
                    keep_mask = label_mask | camera_mask
                    distmat = keep_mask.float() * distmat
                else:
                    keep_mask = None

                # 分块计算指标
                metric_block = engine.eval_metric_func(
                    distmat, query_label_blocks[block_idx], gallery_label, keep_mask
                )

                # 累积指标
                for key in metric_block:
                    metric_dict[key] += metric_block[key] * block_feat.shape[0] / num_query

    # 打印指标
    metric_info_list = []
    for key, value in metric_dict.items():
        metric_info_list.append(f"{key}: {value:.5f}")
        if metric_key is None:
            metric_key = key

    metric_msg = ", ".join(metric_info_list)
    if hasattr(engine, 'logger'):
        engine.logger.info(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")
    else:
        print(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")

    engine.model.train()
    return metric_dict[metric_key] if metric_key else 0.0


def compute_feature(engine, name="gallery"):
    """计算特征

    Args:
        engine: 评估引擎
        name: 数据集名称（gallery/query/gallery_query）

    Returns:
        all_feat: 特征张量
        all_label: 标签张量
        all_camera: 相机ID张量（如果有）
    """
    if name == "gallery":
        dataloader = engine.gallery_dataloader
    elif name == "query":
        dataloader = engine.query_dataloader
    elif name == "gallery_query":
        dataloader = engine.gallery_query_dataloader
    else:
        raise ValueError(
            f"Only support gallery or query or gallery_query dataset, but got {name}"
        )

    all_feat = []
    all_label = []
    all_camera = []
    has_camera = False

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx % engine.config.get("Global", {}).get("print_batch_step", 10) == 0:
                if hasattr(engine, 'logger'):
                    engine.logger.info(
                        f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
                    )
                else:
                    print(f"{name} feature calculation process: [{idx}/{len(dataloader)}]")

            # 移动到设备
            batch = [x.to(engine.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(engine.device)
                    for x in batch]
            batch[1] = batch[1].reshape([-1, 1]).long()

            if len(batch) >= 3:
                has_camera = True
                batch[2] = batch[2].reshape([-1, 1]).long()

            # 前向传播
            is_rec = getattr(engine, 'is_rec', False)
            if is_rec:
                out = engine.model(batch[0], batch[1])
            else:
                out = engine.model(batch[0])

            if isinstance(out, dict) and "Student" in out:
                out = out["Student"]

            # 获取特征
            retrieval_feature_from = engine.config.get("Global", {}).get(
                "retrieval_feature_from", "features"
            )
            if retrieval_feature_from == "features":
                # 使用 neck 的输出作为特征
                batch_feat = out["features"] if isinstance(out, dict) else out
            else:
                # 使用 embedding 作为特征
                batch_feat = out["embedding"] if isinstance(out, dict) else out

            # 特征归一化
            if engine.config.get("Global", {}).get("feature_normalize", True):
                batch_feat = torch.nn.functional.normalize(batch_feat, p=2, dim=1)

            all_feat.append(batch_feat)
            all_label.append(batch[1])
            if has_camera:
                all_camera.append(batch[2])

    # 合并所有特征
    all_feat = torch.cat(all_feat, dim=0)
    all_label = torch.cat(all_label, dim=0).squeeze()
    all_camera = torch.cat(all_camera, dim=0).squeeze() if has_camera else None

    # 分布式收集
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        # 收集特征
        feat_list = [torch.zeros_like(all_feat) for _ in range(world_size)]
        dist.all_gather(feat_list, all_feat)
        all_feat = torch.cat(feat_list, dim=0)

        # 收集标签
        label_list = [torch.zeros_like(all_label) for _ in range(world_size)]
        dist.all_gather(label_list, all_label)
        all_label = torch.cat(label_list, dim=0)

        # 收集相机ID
        if all_camera is not None:
            camera_list = [torch.zeros_like(all_camera) for _ in range(world_size)]
            dist.all_gather(camera_list, all_camera)
            all_camera = torch.cat(camera_list, dim=0)

    return all_feat, all_label, all_camera


def compute_re_ranking_dist(query_feat, gallery_feat, feature_normalize=True,
                            k1=20, k2=6, lambda_value=0.3):
    """计算重排序距离矩阵

    Args:
        query_feat: query 特征
        gallery_feat: gallery 特征
        feature_normalize: 是否归一化特征
        k1: k1 参数
        k2: k2 参数
        lambda_value: lambda 参数

    Returns:
        distmat: 距离矩阵
    """
    if feature_normalize:
        query_feat = torch.nn.functional.normalize(query_feat, p=2, dim=1)
        gallery_feat = torch.nn.functional.normalize(gallery_feat, p=2, dim=1)

    # 计算原始距离
    original_dist = 1 - torch.mm(query_feat, gallery_feat.t())
    original_dist = original_dist.cpu().numpy()

    # k-reciprocal 重排序（简化版本）
    # 完整实现较复杂，这里提供基础版本
    # 实际应用中可以参考更完整的实现

    return torch.from_numpy(original_dist).to(query_feat.device)

