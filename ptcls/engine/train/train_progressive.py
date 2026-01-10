# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Migrated to PyTorch for visionhub project.

from ...data import build_dataloader
from .utils import type_name
from .train import train_epoch


def train_epoch_progressive(engine, epoch_id, print_batch_step):
    """渐进式训练一个 epoch

    渐进式训练通过逐步增加训练难度来提升模型性能：
    - 逐步增大输入图像尺寸
    - 逐步增强数据增强强度
    - 逐步增加 dropout 率

    Args:
        engine: 训练引擎
        epoch_id: 当前 epoch ID
        print_batch_step: 打印间隔步数
    """
    # 1. 构建不同训练阶段的超参数
    num_stage = 4
    ratio_list = [(i + 1) / num_stage for i in range(num_stage)]
    stones = [
        int(engine.config["Global"]["epochs"] * ratio_list[i])
        for i in range(num_stage)
    ]
    stage_id = 0
    for i in range(num_stage):
        if epoch_id > stones[i]:
            stage_id = i + 1

    # 2. 调整不同训练阶段的超参数
    cur_dropout_rate = 0.0
    cur_image_size = 224
    cur_magnitude = 0

    if not hasattr(engine, 'last_stage') or engine.last_stage < stage_id:

        def _change_dp_func(m):
            nonlocal cur_dropout_rate
            if type_name(m) == "Head" and hasattr(m, "_dropout"):
                m._dropout.p = m.dropout_rate[stage_id]
                cur_dropout_rate = m.dropout_rate[stage_id]

        engine.model.apply(_change_dp_func)

        # 更新图像尺寸和数据增强强度
        cur_image_size = engine.config["DataLoader"]["Train"]["dataset"][
            "transform_ops"][1]["RandCropImage"]["progress_size"][stage_id]
        cur_magnitude = engine.config["DataLoader"]["Train"]["dataset"][
            "transform_ops"][3]["RandAugmentV2"]["progress_magnitude"][stage_id]

        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][1][
            "RandCropImage"]["size"] = cur_image_size
        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][3][
            "RandAugmentV2"]["magnitude"] = cur_magnitude

        # 重新构建数据加载器
        engine.train_dataloader = build_dataloader(
            engine.config["DataLoader"],
            "Train",
            engine.device,
            seed=epoch_id
        )
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.last_stage = stage_id

    if hasattr(engine, 'logger'):
        engine.logger.info(
            f"Training stage: [{stage_id+1}/{num_stage}]("
            f"random_aug_magnitude={cur_magnitude}, "
            f"train_image_size={cur_image_size}, "
            f"dropout_rate={cur_dropout_rate}"
            f")"
        )
    else:
        print(
            f"Training stage: [{stage_id+1}/{num_stage}]("
            f"random_aug_magnitude={cur_magnitude}, "
            f"train_image_size={cur_image_size}, "
            f"dropout_rate={cur_dropout_rate}"
            f")"
        )

    # 3. 在当前阶段正常训练一个 epoch
    train_epoch(engine, epoch_id, print_batch_step)

