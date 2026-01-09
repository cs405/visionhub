from .ops.operators import DecodeImage, ResizeImage, CropImage, NormalizeImage, ToCHWImage, ToTensor
from .ops.augmentation import (
    RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation,
    RandomGrayscale, RandomGaussianBlur, RandomErasing, RandomResizedCrop
)

def transform(data, ops=[]):
    for op in ops:
        data = op(data)
    return data

class create_operators(object):
    def __init__(self, op_configs):
        self.ops = []
        for config in op_configs:
            assert isinstance(config, dict) and len(config) == 1
            op_name = list(config.keys())[0]
            param = config[op_name]
            if param is None:
                param = {}
            op = eval(op_name)(**param)
            self.ops.append(op)

    def __call__(self, data):
        return transform(data, self.ops)
