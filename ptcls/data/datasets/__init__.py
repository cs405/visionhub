"""__init__.py for datasets module"""

from .common_dataset import CommonDataset, ImageNetDataset, MultiLabelDataset
from .vehicle_logo_dataset import CompCarsDataset, VeriWildDataset, LogoDataset
from .person_face_dataset import PersonDataset, FaceDataset, CustomLabelDataset
from .cifar_dataset import CIFARDataset, CIFAR10Dataset, CIFAR100Dataset
from .niche_datasets import ProductDataset, FoodDataset, LandmarkDataset

__all__ = [
    'CommonDataset',
    'ImageNetDataset',
    'MultiLabelDataset',
    'CompCarsDataset',
    'VeriWildDataset',
    'LogoDataset',
    'PersonDataset',
    'FaceDataset',
    'CustomLabelDataset',
    'CIFARDataset',
    'CIFAR10Dataset',
    'CIFAR100Dataset',
    'ProductDataset',
    'FoodDataset',
    'LandmarkDataset',
]

