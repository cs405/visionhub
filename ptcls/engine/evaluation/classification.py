import time
import torch
import torch.distributed as dist
from ...utils.avgmeter import AverageMeter


def classification_eval(engine, epoch_id=0):
    """分类模型评估

    Args:
        engine: 评估引擎
        epoch_id: epoch ID

    Returns:
        metric_msg: 指标信息字符串
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
            batch_size = batch[0].shape[0]

            # 移动到设备
            batch[0] = batch[0].to(engine.device)
            if not engine.config.get("Global", {}).get("use_multilabel", False):
                if len(batch[1].shape) == 1:
                    batch[1] = batch[1].reshape([-1, 1])
            batch[1] = batch[1].to(engine.device)

            # 前向传播
            is_rec = getattr(engine, 'is_rec', False)
            if is_rec:
                out = engine.model(batch[0], batch[1])
            else:
                out = engine.model(batch[0])

            # 分布式采样问题：重复采样
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            current_samples = batch_size * world_size
            accum_samples += current_samples

            # 处理输出（蒸馏模型等）
            if isinstance(out, dict) and "Student" in out:
                out = out["Student"]
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]

            # 分布式收集
            if dist.is_initialized() and world_size > 1:
                label_list = [torch.zeros_like(batch[1]) for _ in range(world_size)]
                dist.all_gather(label_list, batch[1])
                labels = torch.cat(label_list, 0)

                if isinstance(out, list):
                    preds = []
                    for x in out:
                        pred_list = [torch.zeros_like(x) for _ in range(world_size)]
                        dist.all_gather(pred_list, x)
                        pred_x = torch.cat(pred_list, 0)
                        preds.append(pred_x)
                else:
                    pred_list = [torch.zeros_like(out) for _ in range(world_size)]
                    dist.all_gather(pred_list, out)
                    preds = torch.cat(pred_list, 0)

                # 处理重复采样
                if accum_samples > total_samples:
                    trim_size = total_samples + current_samples - accum_samples
                    if isinstance(preds, list):
                        preds = [pred[:trim_size] for pred in preds]
                    else:
                        preds = preds[:trim_size]
                    labels = labels[:trim_size]
                    current_samples = trim_size
            else:
                labels = batch[1]
                preds = out

            # 计算损失
            if engine.eval_loss_func is not None:
                loss_dict = engine.eval_loss_func(preds, labels)
                for key in loss_dict:
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(float(loss_dict[key]), current_samples)

            # 计算指标
            if engine.eval_metric_func is not None:
                engine.eval_metric_func(preds, labels)

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

                if hasattr(engine.eval_metric_func, 'avg_info'):
                    metric_msg += f", {engine.eval_metric_func.avg_info}"

                log_msg = (f"[Eval][Epoch {epoch_id}][Iter: {iter_id}/{len(engine.eval_dataloader)}]"
                          f"{metric_msg}, {time_msg}, {ips_msg}")

                if hasattr(engine, 'logger'):
                    engine.logger.info(log_msg)
                else:
                    print(log_msg)

            tic = time.time()

    # 获取最终指标
    metric_msg = ""
    if engine.eval_metric_func is not None:
        if hasattr(engine.eval_metric_func, 'avg_info'):
            metric_msg = engine.eval_metric_func.avg_info
        elif hasattr(engine.eval_metric_func, 'avg'):
            metric_msg = f"Metric: {engine.eval_metric_func.avg():.5f}"

    # 添加损失信息
    if output_info:
        loss_msg = ", ".join([
            f"{key}: {output_info[key].avg:.5f}"
            for key in output_info
        ])
        if metric_msg:
            metric_msg = f"{loss_msg}, {metric_msg}"
        else:
            metric_msg = loss_msg

    if hasattr(engine, 'logger'):
        engine.logger.info(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")
    else:
        print(f"[Eval][Epoch {epoch_id}][Avg]{metric_msg}")

    # 恢复训练模式
    engine.model.train()

    return metric_msg

