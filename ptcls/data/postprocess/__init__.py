# Migrated to PyTorch for visionhub project.

import copy
import importlib

from .topk import Topk
from .threshoutput import ThreshOutput, MultiLabelThreshOutput
from .scoreoutput import ScoreOutput
from .attribute import PersonAttribute, VehicleAttribute, FaceAttribute, TableAttribute
from .binarize import Binarize
from .save_prelabel import SavePreLabel

__all__ = [
    'Topk',
    'ThreshOutput',
    'MultiLabelThreshOutput',
    'ScoreOutput',
    'PersonAttribute',
    'VehicleAttribute',
    'FaceAttribute',
    'TableAttribute',
    'Binarize',
    'SavePreLabel',
    'build_postprocess',
    'DistillationPostProcess'
]


def build_postprocess(config):
    """构建后处理模块

    Args:
        config: 后处理配置字典

    Returns:
        后处理函数对象
    """
    if config is None:
        return None

    config = copy.deepcopy(config)
    model_name = config.pop("name")
    mod = importlib.import_module(__name__)
    postprocess_func = getattr(mod, model_name)(**config)
    return postprocess_func


class DistillationPostProcess(object):
    """蒸馏模型后处理"""

    def __init__(self, model_name="Student", key=None, func="Topk", **kwargs):
        super().__init__()
        self.func = eval(func)(**kwargs)
        self.model_name = model_name
        self.key = key

    def __call__(self, x, file_names=None):
        x = x[self.model_name]
        if self.key is not None:
            x = x[self.key]
        return self.func(x, file_names=file_names)

