# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

import time
import torch
import torch.nn.functional as F
from .utils import update_loss, update_metric, log_info


def train_epoch_fixmatch(engine, epoch_id, print_batch_step):
    """FixMatch 半监督训练一个 epoch

    FixMatch 是一种半监督学习方法，结合一致性正则化和伪标签。

    Args:
        engine: 训练引擎
        epoch_id: 当前 epoch ID
        print_batch_step: 打印间隔步数
    """
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.unlabel_train_dataloader_iter = iter(
            engine.unlabel_train_dataloader)

    # FixMatch 超参数
    temperature = engine.config.get("SSL", {}).get("temperature", 1)
    threshold = engine.config.get("SSL", {}).get("threshold", 0.95)

    assert hasattr(engine, 'iter_per_epoch') and engine.iter_per_epoch is not None, \
        "Global.iter_per_epoch need to be set."

    threshold = torch.tensor(threshold, device=engine.device)

    for iter_id in range(engine.iter_per_epoch):
        if iter_id >= engine.iter_per_epoch:
            break

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()

        # 获取有标签数据
        try:
            label_data_batch = next(engine.train_dataloader_iter)
        except StopIteration:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            label_data_batch = next(engine.train_dataloader_iter)

        # 获取无标签数据
        try:
            unlabel_data_batch = next(engine.unlabel_train_dataloader_iter)
        except StopIteration:
            engine.unlabel_train_dataloader_iter = iter(
                engine.unlabel_train_dataloader)
            unlabel_data_batch = next(engine.unlabel_train_dataloader_iter)

        assert len(unlabel_data_batch) == 3, \
            "Unlabeled data should have 3 elements: weak_aug, strong_aug, target"
        assert unlabel_data_batch[0].shape == unlabel_data_batch[1].shape, \
            "Weak and strong augmented images should have same shape"

        engine.time_info["reader_cost"].update(time.time() - tic)

        batch_size = (label_data_batch[0].shape[0] +
                     unlabel_data_batch[0].shape[0] +
                     unlabel_data_batch[1].shape[0])
        engine.global_step += 1

        # 准备输入
        inputs_x, targets_x = label_data_batch
        inputs_u_w, inputs_u_s, targets_u = unlabel_data_batch
        batch_size_label = inputs_x.shape[0]

        # 合并输入
        inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)

        # 前向传播（使用混合精度）
        if hasattr(engine, 'scaler') and engine.scaler is not None:
            with torch.cuda.amp.autocast(enabled=engine.use_amp):
                loss_dict, logits_label = get_loss(
                    engine, inputs, batch_size_label,
                    temperature, threshold, targets_x
                )

            loss = loss_dict["loss"]

            # 反向传播
            engine.scaler.scale(loss).backward()

            # 优化器步骤
            if isinstance(engine.optimizer, list):
                for opt in engine.optimizer:
                    engine.scaler.step(opt)
            else:
                engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            loss_dict, logits_label = get_loss(
                engine, inputs, batch_size_label,
                temperature, threshold, targets_x
            )

            loss = loss_dict["loss"]
            loss.backward()

            if isinstance(engine.optimizer, list):
                for opt in engine.optimizer:
                    opt.step()
            else:
                engine.optimizer.step()

        # 更新学习率（按步）
        if hasattr(engine, 'lr_scheduler'):
            lr_sch_list = (engine.lr_scheduler if isinstance(engine.lr_scheduler, list)
                          else [engine.lr_scheduler])
            for lr_sch in lr_sch_list:
                if lr_sch is not None and not getattr(lr_sch, "by_epoch", False):
                    lr_sch.step()

        # 清除梯度
        if isinstance(engine.optimizer, list):
            for opt in engine.optimizer:
                opt.zero_grad()
        else:
            engine.optimizer.zero_grad()

        # 更新 EMA
        if hasattr(engine, 'ema') and engine.ema is not None:
            engine.ema.update(engine.model)

        # 日志记录（只记录有标签数据的指标）
        update_metric(engine, logits_label, label_data_batch, batch_size)
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)

        tic = time.time()

    # 更新学习率（按 epoch）
    if hasattr(engine, 'lr_scheduler'):
        lr_sch_list = (engine.lr_scheduler if isinstance(engine.lr_scheduler, list)
                      else [engine.lr_scheduler])
        for lr_sch in lr_sch_list:
            if lr_sch is not None and getattr(lr_sch, "by_epoch", False):
                lr_sch.step()


def get_loss(engine, inputs, batch_size_label, temperature, threshold, targets_x):
    """计算 FixMatch 损失

    Args:
        engine: 训练引擎
        inputs: 合并的输入（有标签 + 弱增强无标签 + 强增强无标签）
        batch_size_label: 有标签数据的批次大小
        temperature: 温度参数
        threshold: 伪标签阈值
        targets_x: 有标签数据的目标

    Returns:
        loss_dict: 损失字典
        logits_x: 有标签数据的 logits
    """
    # 前向传播
    logits = engine.model(inputs)

    # 分离 logits
    logits_x = logits[:batch_size_label]
    logits_u_w, logits_u_s = logits[batch_size_label:].chunk(2)

    # 有标签数据的损失
    loss_dict_label = engine.train_loss_func(logits_x, targets_x)

    # 无标签数据的伪标签
    with torch.no_grad():
        probs_u_w = F.softmax(logits_u_w.detach() / temperature, dim=-1)

    p_targets_u, mask = get_pseudo_label_and_mask(probs_u_w, threshold)

    # 无标签数据的损失
    if hasattr(engine, 'unlabel_train_loss_func'):
        unlabel_celoss = engine.unlabel_train_loss_func(logits_u_s, p_targets_u)["CELoss"]
    else:
        # 如果没有专门的无标签损失函数，使用交叉熵
        unlabel_celoss = F.cross_entropy(logits_u_s, p_targets_u, reduction='none')

    unlabel_celoss = (unlabel_celoss * mask).mean()

    # 合并损失
    loss_dict = {}
    for k, v in loss_dict_label.items():
        if k != "loss":
            loss_dict[k + "_label"] = v

    loss_dict["CELoss_unlabel"] = unlabel_celoss
    loss_dict["loss"] = loss_dict_label['loss'] + unlabel_celoss

    return loss_dict, logits_x


def get_pseudo_label_and_mask(probs_u_w, threshold):
    """生成伪标签和mask

    Args:
        probs_u_w: 弱增强无标签数据的概率
        threshold: 置信度阈值

    Returns:
        p_targets_u: 伪标签
        mask: 置信度mask
    """
    max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
    mask = (max_probs >= threshold).float()

    return p_targets_u, mask

