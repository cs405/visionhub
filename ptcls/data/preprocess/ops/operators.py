import cv2
import numpy as np
from PIL import Image
import torch

class DecodeImage(object):
    def __init__(self, to_np=True, to_rgb=True, channel_first=False, backend="cv2"):
        self.to_np = to_np
        self.to_rgb = to_rgb
        self.channel_first = channel_first
        self.backend = backend

    def __call__(self, img):
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = f.read()
        
        if isinstance(img, bytes):
            if self.backend == 'cv2':
                img = np.frombuffer(img, dtype='uint8')
                img = cv2.imdecode(img, 1) # BGR
            else:
                img = Image.open(io.BytesIO(img)).convert('RGB')
                img = np.array(img)

        if self.to_rgb and self.backend == 'cv2':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.channel_first:
            img = img.transpose((2, 0, 1))
            
        if self.to_np:
            img = np.array(img)
            
        return img

class ResizeImage(object):
    def __init__(self, size=None, resize_short=None, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.resize_short = resize_short
        self.interpolation = interpolation

    def __call__(self, img):
        if self.size is not None:
            if isinstance(self.size, int):
                size = (self.size, self.size)
            else:
                size = self.size
            return cv2.resize(img, size, interpolation=self.interpolation)
        
        if self.resize_short is not None:
            h, w = img.shape[:2]
            if h > w:
                new_w = self.resize_short
                new_h = int(h * (self.resize_short / w))
            else:
                new_h = self.resize_short
                new_w = int(w * (self.resize_short / h))
            return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
        return img

class CropImage(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2
        return img[h_start:h_start + h, w_start:w_start + w, :]

class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        # yaml 里常写 1.0/255.0，可能被当成字符串
        if isinstance(scale, str):
            expr = scale.strip()
            # minimal safe eval: only digits/operators
            allowed = set('0123456789.+-*/() ')
            if not set(expr) <= allowed:
                raise ValueError(f"Unsafe scale expression: {scale}")
            try:
                scale = float(eval(expr, {"__builtins__": {}}, {}))
            except Exception as e:
                raise ValueError(f"Invalid scale: {scale}") from e
        self.scale = float(scale) if scale is not None else None

        self.mean = np.array(mean).reshape((1, 1, 3)).astype('float32') if mean else None
        self.std = np.array(std).reshape((1, 1, 3)).astype('float32') if std else None
        self.order = order

    def __call__(self, img):
        img = img.astype('float32')
        if self.scale is not None:
            img *= self.scale
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std

        if self.order == 'chw':
            img = img.transpose((2, 0, 1))
        return img

class ToCHWImage(object):
    def __call__(self, img):
        if len(img.shape) == 3:
            return img.transpose((2, 0, 1))
        return img

class ToTensor(object):
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return torch.from_numpy(img)
        return img
