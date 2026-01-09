import os
import copy
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial

# NOTE: 当前版本 ImageNetDataset 定义在 common_dataset.py
from .dataloader.common_dataset import ImageNetDataset
from .preprocess import create_operators, transform
from ..utils import logger

def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(config, mode, device, seed=None):
    assert mode in ['Train', 'Eval', 'Test'], "Dataset mode error"
    assert mode in config.keys(), "{} config not in yaml".format(mode)
    
    config_dataset = copy.deepcopy(config[mode]['dataset'])
    dataset_name = config_dataset.pop('name')
    dataset = eval(dataset_name)(**config_dataset)
    
    config_sampler = config[mode]['sampler']
    batch_size = config_sampler.get("batch_size", 1)
    drop_last = config_sampler.get("drop_last", False)
    shuffle = config_sampler.get("shuffle", False)
    
    # 简单的分布式支持
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    config_loader = config[mode]['loader']
    num_workers = config_loader.get("num_workers", 0)
    
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
        worker_init_fn=init_fn,
        pin_memory=True
    )
    
    return data_loader
