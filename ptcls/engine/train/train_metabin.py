import time
import torch
import numpy as np
from collections import defaultdict

from .utils import update_loss, update_metric, log_info, type_name
from ...data import build_dataloader
from ...loss import build_loss


def train_epoch_metabin(engine, epoch_id, print_batch_step):
    """MetaBIN 元学习训练一个 epoch

    MetaBIN (Meta Batch-Instance Normalization) 是一种元学习方法，
    通过元训练和元测试来学习域不变特征。

    Args:
        engine: 训练引擎
        epoch_id: 当前 epoch ID
        print_batch_step: 打印间隔步数
    """
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)

    if not hasattr(engine, "meta_dataloader"):
        engine.meta_dataloader = build_dataloader(
            config=engine.config['DataLoader']['Metalearning'],
            mode='Train',
            device=engine.device
        )
        engine.meta_dataloader_iter = iter(engine.meta_dataloader)

    num_domain = engine.train_dataloader.dataset.num_cams

    for iter_id in range(engine.iter_per_epoch):
        # 获取训练数据批次
        try:
            train_batch = next(engine.train_dataloader_iter)
        except StopIteration:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            train_batch = next(engine.train_dataloader_iter)

        # 获取元学习数据
        try:
            mtrain_batch, mtest_batch = get_meta_data(
                engine.meta_dataloader_iter, num_domain
            )
        except StopIteration:
            engine.meta_dataloader_iter = iter(engine.meta_dataloader)
            mtrain_batch, mtest_batch = get_meta_data(
                engine.meta_dataloader_iter, num_domain
            )

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()

        engine.time_info["reader_cost"].update(time.time() - tic)

        train_batch_size = train_batch[0].shape[0]
        mtrain_batch_size = mtrain_batch[0].shape[0]
        mtest_batch_size = mtest_batch[0].shape[0]

        if not engine.config.get("Global", {}).get("use_multilabel", False):
            train_batch[1] = train_batch[1].reshape([train_batch_size, -1])
            mtrain_batch[1] = mtrain_batch[1].reshape([mtrain_batch_size, -1])
            mtest_batch[1] = mtest_batch[1].reshape([mtest_batch_size, -1])

        engine.global_step += 1

        # Warmup: 更新模型（除 gate 外）
        if engine.global_step == 1:
            warmup_iter = engine.config.get("Global", {}).get("warmup_iter", 1) - 1
            for i in range(warmup_iter):
                out, basic_loss_dict = basic_update(engine, train_batch)
                loss_dict = basic_loss_dict
                try:
                    train_batch = next(engine.train_dataloader_iter)
                except StopIteration:
                    engine.train_dataloader_iter = iter(engine.train_dataloader)
                    train_batch = next(engine.train_dataloader_iter)

        # 基础更新
        out, basic_loss_dict = basic_update(engine=engine, batch=train_batch)

        # 元学习更新
        mtrain_loss_dict, mtest_loss_dict = metalearning_update(
            engine=engine, mtrain_batch=mtrain_batch, mtest_batch=mtest_batch
        )

        # 合并损失字典
        loss_dict = {
            **{"train_" + key: value for key, value in basic_loss_dict.items()},
            **{"mtrain_" + key: value for key, value in mtrain_loss_dict.items()},
            **{"mtest_" + key: value for key, value in mtest_loss_dict.items()}
        }

        # 更新学习率（按步）
        if hasattr(engine, 'lr_scheduler'):
            lr_sch_list = (engine.lr_scheduler if isinstance(engine.lr_scheduler, list)
                          else [engine.lr_scheduler])
            for lr_sch in lr_sch_list:
                if lr_sch is not None and not getattr(lr_sch, "by_epoch", False):
                    lr_sch.step()

        # 更新 EMA
        if hasattr(engine, 'ema') and engine.ema is not None:
            engine.ema.update(engine.model)

        # 日志记录
        update_metric(engine, out, train_batch, train_batch_size)
        update_loss(engine, loss_dict, train_batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            log_info(engine, train_batch_size, epoch_id, iter_id)

        tic = time.time()

    # 更新学习率（按 epoch）
    if hasattr(engine, 'lr_scheduler'):
        lr_sch_list = (engine.lr_scheduler if isinstance(engine.lr_scheduler, list)
                      else [engine.lr_scheduler])
        for lr_sch in lr_sch_list:
            if lr_sch is not None and getattr(lr_sch, "by_epoch", False):
                if type_name(lr_sch) != "ReduceLROnPlateau":
                    lr_sch.step()


def setup_opt(engine, stage):
    """设置优化选项

    Args:
        engine: 训练引擎
        stage: 训练阶段（train/mtrain/mtest）
    """
    assert stage in ["train", "mtrain", "mtest"]
    opt = defaultdict()

    if stage == "train":
        opt["bn_mode"] = "general"
        opt["enable_inside_update"] = False
        opt["lr_gate"] = 0.0
    elif stage == "mtrain":
        opt["bn_mode"] = "hold"
        opt["enable_inside_update"] = False
        opt["lr_gate"] = 0.0
    elif stage == "mtest":
        # 获取学习率
        if isinstance(engine.lr_scheduler, list):
            norm_lr = engine.lr_scheduler[1].get_last_lr()[0]
            cyclic_lr = engine.lr_scheduler[2].get_last_lr()[0]
        else:
            norm_lr = engine.lr_scheduler.get_last_lr()[0]
            cyclic_lr = 1.0

        opt["bn_mode"] = "hold"
        opt["enable_inside_update"] = True
        opt["lr_gate"] = norm_lr * cyclic_lr

    # 设置 MetaBIN 层选项
    for layer in engine.model.modules():
        if type_name(layer) == "MetaBIN":
            if hasattr(layer, 'setup_opt'):
                layer.setup_opt(opt)

    if hasattr(engine.model, 'neck') and hasattr(engine.model.neck, 'setup_opt'):
        engine.model.neck.setup_opt(opt)


def reset_opt(model):
    """重置优化选项

    Args:
        model: 模型
    """
    for layer in model.modules():
        if type_name(layer) == "MetaBIN":
            if hasattr(layer, 'reset_opt'):
                layer.reset_opt()

    if hasattr(model, 'neck') and hasattr(model.neck, 'reset_opt'):
        model.neck.reset_opt()


def get_meta_data(meta_dataloader_iter, num_domain):
    """获取元学习数据并按域划分

    Args:
        meta_dataloader_iter: 元数据加载器迭代器
        num_domain: 域的数量

    Returns:
        mtrain_batch: 元训练批次
        mtest_batch: 元测试批次
    """
    list_all = np.random.permutation(num_domain)
    list_mtrain = list(list_all[:num_domain // 2])
    batch = next(meta_dataloader_iter)
    domain_idx = batch[2]

    # 构建元训练域 mask
    is_mtrain_domain = torch.zeros_like(domain_idx, dtype=torch.bool)
    for sample in list_mtrain:
        is_mtrain_domain = torch.logical_or(is_mtrain_domain, domain_idx == sample)

    # 元训练批次
    if not torch.any(is_mtrain_domain):
        raise RuntimeError("No meta-train samples found")
    mtrain_batch = [batch[i][is_mtrain_domain] for i in range(len(batch))]

    # 元测试批次
    is_mtest_domains = ~is_mtrain_domain
    if not torch.any(is_mtest_domains):
        raise RuntimeError("No meta-test samples found")
    mtest_batch = [batch[i][is_mtest_domains] for i in range(len(batch))]

    return mtrain_batch, mtest_batch


def forward(engine, batch, loss_func):
    """前向传播

    Args:
        engine: 训练引擎
        batch: 数据批次
        loss_func: 损失函数

    Returns:
        out: 模型输出
        loss_dict: 损失字典
    """
    batch_info = {"label": batch[1], "domain": batch[2]}

    if hasattr(engine, 'scaler') and engine.scaler is not None:
        with torch.cuda.amp.autocast(enabled=engine.use_amp):
            out = engine.model(batch[0], batch[1])
            loss_dict = loss_func(out, batch_info)
    else:
        out = engine.model(batch[0], batch[1])
        loss_dict = loss_func(out, batch_info)

    return out, loss_dict


def backward(engine, loss, optimizer):
    """反向传播

    Args:
        engine: 训练引擎
        loss: 损失值
        optimizer: 优化器
    """
    optimizer.zero_grad()

    if hasattr(engine, 'scaler') and engine.scaler is not None:
        engine.scaler.scale(loss).backward()
        engine.scaler.step(optimizer)
        engine.scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Clip gate 参数
    for name, module in engine.model.named_modules():
        if "gate" == name.split('.')[-1]:
            if hasattr(module, 'clip_gate'):
                module.clip_gate()


def basic_update(engine, batch):
    """基础更新

    Args:
        engine: 训练引擎
        batch: 数据批次

    Returns:
        out: 模型输出
        train_loss_dict: 训练损失字典
    """
    setup_opt(engine, "train")
    train_loss_func = build_loss(engine.config["Loss"]["Basic"])
    out, train_loss_dict = forward(engine, batch, train_loss_func)
    train_loss = train_loss_dict["loss"]

    if isinstance(engine.optimizer, list):
        backward(engine, train_loss, engine.optimizer[0])
        engine.optimizer[0].zero_grad()
    else:
        backward(engine, train_loss, engine.optimizer)
        engine.optimizer.zero_grad()

    reset_opt(engine.model)
    return out, train_loss_dict


def metalearning_update(engine, mtrain_batch, mtest_batch):
    """元学习更新

    Args:
        engine: 训练引擎
        mtrain_batch: 元训练批次
        mtest_batch: 元测试批次

    Returns:
        mtrain_loss_dict: 元训练损失字典
        mtest_loss_dict: 元测试损失字典
    """
    # 元训练
    mtrain_loss_func = build_loss(engine.config["Loss"]["MetaTrain"])
    setup_opt(engine, "mtrain")

    mtrain_batch_info = {"label": mtrain_batch[1], "domain": mtrain_batch[2]}
    out = engine.model(mtrain_batch[0], mtrain_batch[1])
    mtrain_loss_dict = mtrain_loss_func(out, mtrain_batch_info)
    mtrain_loss = mtrain_loss_dict["loss"]

    optimizer_idx = 1 if isinstance(engine.optimizer, list) else 0
    optimizer = engine.optimizer[optimizer_idx] if isinstance(engine.optimizer, list) else engine.optimizer

    optimizer.zero_grad()
    mtrain_loss.backward()

    # 元测试
    mtest_loss_func = build_loss(engine.config["Loss"]["MetaTest"])
    setup_opt(engine, "mtest")

    out, mtest_loss_dict = forward(engine, mtest_batch, mtest_loss_func)
    optimizer.zero_grad()
    mtest_loss = mtest_loss_dict["loss"]
    backward(engine, mtest_loss, optimizer)

    # 清除梯度
    if isinstance(engine.optimizer, list):
        engine.optimizer[0].zero_grad()
        engine.optimizer[1].zero_grad()
    else:
        engine.optimizer.zero_grad()

    reset_opt(engine.model)

    return mtrain_loss_dict, mtest_loss_dict

