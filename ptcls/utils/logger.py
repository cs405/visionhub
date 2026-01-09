import datetime
import logging
import os
import sys
import torch.distributed as dist

_logger = None

class LoggerHook(object):
    """
    保证日志只在特定rank输出
    """
    block = False

    def __init__(self, log_func):
        self.log_func = log_func

    def __call__(self, *args, **kwargs):
        if not self.block:
            self.log_func(*args, **kwargs)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def init_logger(name='ptcls',
                log_file=None,
                log_level=logging.INFO,
                log_ranks="0"):
    global _logger

    init_flag = False
    if _logger is None:
        _logger = logging.getLogger(name)
        init_flag = True

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.set_name('stream_handler')

    # 避免重复添加handler
    handler_names = [h.get_name() for h in _logger.handlers]
    if 'stream_handler' not in handler_names:
        _logger.addHandler(stream_handler)

    rank = get_rank()

    if log_file is not None and rank == 0:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        file_handler.set_name('file_handler')

        if 'file_handler' not in [h.get_name() for h in _logger.handlers]:
            _logger.addHandler(file_handler)

    if isinstance(log_ranks, str):
        log_ranks = [int(i) for i in log_ranks.split(',')]
    elif isinstance(log_ranks, int):
        log_ranks = [log_ranks]
    
    if rank in log_ranks:
        _logger.setLevel(log_level)
        LoggerHook.block = False
    else:
        _logger.setLevel(logging.ERROR)
        LoggerHook.block = True
    _logger.propagate = False

@LoggerHook
def info(fmt, *args):
    if _logger:
        _logger.info(fmt, *args)

@LoggerHook
def debug(fmt, *args):
    if _logger:
        _logger.debug(fmt, *args)

@LoggerHook
def warning(fmt, *args):
    if _logger:
        _logger.warning(fmt, *args)

@LoggerHook
def error(fmt, *args):
    if _logger:
        _logger.error(fmt, *args)

def advertise():
    copyright = "visionhub is a PyTorch port of visionhub !"
    ad = "For more info please go to the original visionhub repo."
    website = "https://github.com/visionhubvisionhub/visionhub"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    info("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
        "=" * (AD_LEN + 4),
        "=={}==".format(copyright.center(AD_LEN)),
        "=" * (AD_LEN + 4),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(ad.center(AD_LEN)),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(website.center(AD_LEN)),
        "=" * (AD_LEN + 4), ))
