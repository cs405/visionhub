import time
import torch
from .utils import update_loss, update_metric, log_info, type_name


def train_epoch(engine, epoch_id, print_batch_step):
    """标准训练一个 epoch

    Args:
        engine: 训练引擎
        epoch_id: 当前 epoch ID
        print_batch_step: 打印间隔步数
    """
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)

    for iter_id in range(engine.iter_per_epoch):
        # 获取数据批次
        try:
            batch = next(engine.train_dataloader_iter)
        except StopIteration:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            batch = next(engine.train_dataloader_iter)

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()

        engine.time_info["reader_cost"].update(time.time() - tic)

        batch_size = batch[0].shape[0]
        if not engine.config.get("Global", {}).get("use_multilabel", False):
            if len(batch[1].shape) == 1:
                batch[1] = batch[1].reshape([batch_size, -1])

        engine.global_step += 1

        # 前向传播
        update_freq = getattr(engine, 'update_freq', 1)

        # 使用混合精度
        if hasattr(engine, 'scaler') and engine.scaler is not None:
            with torch.cuda.amp.autocast(enabled=engine.use_amp):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])

            loss = loss_dict["loss"] / update_freq

            # 反向传播
            engine.scaler.scale(loss).backward()

            if (iter_id + 1) % update_freq == 0:
                # 优化器步骤
                if isinstance(engine.optimizer, list):
                    for opt in engine.optimizer:
                        engine.scaler.step(opt)
                else:
                    engine.scaler.step(engine.optimizer)
                engine.scaler.update()
        else:
            # 不使用混合精度
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])
            loss = loss_dict["loss"] / update_freq

            # 反向传播
            loss.backward()

            if (iter_id + 1) % update_freq == 0:
                if isinstance(engine.optimizer, list):
                    for opt in engine.optimizer:
                        opt.step()
                else:
                    engine.optimizer.step()

        if (iter_id + 1) % update_freq == 0:
            # 清除梯度
            if isinstance(engine.optimizer, list):
                for opt in engine.optimizer:
                    opt.zero_grad()
            else:
                engine.optimizer.zero_grad()

            # 更新学习率（按步）
            if hasattr(engine, 'lr_scheduler'):
                lr_sch_list = engine.lr_scheduler if isinstance(engine.lr_scheduler, list) else [engine.lr_scheduler]
                for lr_sch in lr_sch_list:
                    if lr_sch is not None and not getattr(lr_sch, "by_epoch", False):
                        lr_sch.step()

            # 更新 EMA
            if hasattr(engine, 'ema') and engine.ema is not None:
                engine.ema.update(engine.model if not hasattr(engine, 'model_ema') else engine.model)

        # 日志记录
        update_metric(engine, out, batch, batch_size)
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)

        tic = time.time()

    # 更新学习率（按 epoch）
    if hasattr(engine, 'lr_scheduler'):
        lr_sch_list = engine.lr_scheduler if isinstance(engine.lr_scheduler, list) else [engine.lr_scheduler]
        for lr_sch in lr_sch_list:
            if lr_sch is not None and getattr(lr_sch, "by_epoch", False):
                # 跳过 ReduceLROnPlateau（需要 metric）
                if type_name(lr_sch) != "ReduceLROnPlateau":
                    lr_sch.step()


def forward(engine, batch):
    """前向传播

    Args:
        engine: 训练引擎
        batch: 数据批次

    Returns:
        模型输出
    """
    is_rec = getattr(engine, 'is_rec', False)

    if not is_rec:
        return engine.model(batch[0])
    else:
        # 识别任务（如人脸识别）需要标签
        return engine.model(batch[0], batch[1])

