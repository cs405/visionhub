# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

from ..utils.avgmeter import AverageMeter


def update_loss(engine, loss_dict, batch_size):
    """更新损失记录

    Args:
        engine: 训练引擎
        loss_dict: 损失字典
        batch_size: 批次大小
    """
    for key in loss_dict:
        if key not in engine.output_info:
            engine.output_info[key] = AverageMeter(key, '7.5f')
        engine.output_info[key].update(float(loss_dict[key]), batch_size)


def update_metric(engine, out, batch, batch_size):
    """更新指标记录

    Args:
        engine: 训练引擎
        out: 模型输出
        batch: 数据批次
        batch_size: 批次大小
    """
    if engine.train_metric_func is not None:
        # 计算指标
        metric_dict = engine.train_metric_func(out, batch[1])
        for key in metric_dict:
            if key not in engine.output_info:
                engine.output_info[key] = AverageMeter(key, '7.5f')
            engine.output_info[key].update(float(metric_dict[key]), batch_size)


def log_info(engine, batch_size, epoch_id, iter_id):
    """打印日志信息

    Args:
        engine: 训练引擎
        batch_size: 批次大小
        epoch_id: epoch ID
        iter_id: iteration ID
    """
    # 获取学习率
    lr_msg = "lr(s): "
    if hasattr(engine, 'optimizer'):
        if isinstance(engine.optimizer, list):
            for idx, opt in enumerate(engine.optimizer):
                lr = opt.param_groups[0]['lr']
                lr_msg += f"{lr:.6f}"
                if idx < len(engine.optimizer) - 1:
                    lr_msg += ", "
        else:
            lr = engine.optimizer.param_groups[0]['lr']
            lr_msg += f"{lr:.6f}"

    # 构建输出信息
    log_str = f"[Train][Epoch {epoch_id}/{engine.config['Global']['epochs']}]"
    log_str += f"[Iter {iter_id}/{engine.iter_per_epoch}] "
    log_str += lr_msg

    # 添加时间信息
    if hasattr(engine, 'time_info'):
        for key in engine.time_info:
            log_str += f", {engine.time_info[key].avg_info}"

    # 添加损失和指标信息
    for key in engine.output_info:
        log_str += f", {engine.output_info[key].avg_info}"

    # 打印日志
    if hasattr(engine, 'logger'):
        engine.logger.info(log_str)
    else:
        print(log_str)


def type_name(obj):
    """获取对象类型名称

    Args:
        obj: 对象

    Returns:
        类型名称字符串
    """
    return type(obj).__name__

