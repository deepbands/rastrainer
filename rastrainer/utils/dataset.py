import math
from paddle.io import Dataset
import numpy as np
import paddleseg.transforms as T
from .raster import Raster
from typing import List, Tuple


class QDataset(Dataset):
    def __init__(self, 
                 image_path: str, 
                 label_path: str, 
                 num_classes: int=2, 
                 transforms: List=[]) -> None:
        self.block_size = 512  # default
        self.images = Raster(image_path)
        self.labels = Raster(label_path)
        self.transforms = transforms
        self.num_classes = num_classes
        self.ignore_index = 255
        rows = math.ceil(self.images.height / self.block_size)
        cols = math.ceil(self.images.width / self.block_size)
        self.loc_list = []
        for r in range(rows):
            for c in range(cols):
                self.loc_list.append((c * self.block_size, r * self.block_size))

    def __getitem__(self, idx: int) -> Tuple:
        loc_start = self.loc_list[idx]
        img = self.images.getArray(loc_start, (self.block_size, self.block_size))
        lab = self.labels.getArray(loc_start, (self.block_size, self.block_size))
        for op in self.transforms:
            img, lab = op(img, lab)
        img = np.transpose(img, (2, 0, 1))
        return img, lab

    def __len__(self) -> int:
        return len(self.loc_list)


class QTrainDaraset(QDataset):
    def __init__(self, 
                 image_path: str, 
                 label_path: str, 
                 num_classes: int=2) -> None:
        transforms = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomScaleAspect(),
            T.RandomBlur(),
            T.Resize(target_size=(512, 512)),
            T.Normalize()
        ]
        super().__init__(image_path, label_path, num_classes, transforms)


class QEvalDaraset(QDataset):
    def __init__(self, 
                 image_path: str, 
                 label_path: str, 
                 num_classes: int=2) -> None:
        transforms = [
            T.Resize(target_size=(512, 512)),
            T.Normalize()
        ]
        super().__init__(image_path, label_path, num_classes, transforms)
