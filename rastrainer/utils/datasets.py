import os.path as osp
import numpy as np
import paddleseg.transforms as T
from paddleseg.datasets import Dataset


class InitMask:
    def __init__(self):
        pass

    def __call__(self, im, label=None):
        label = np.clip(label, 0, 1)
        if label is None:
            return (im, )
        else:
            if len(label.shape) == 3:
                label = np.mean(label, axis=-1).astype("uint8")
            return (im, label)


def create_dataset(dataset_root, classes):
    train_transforms = [
        InitMask(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(),
        T.RandomScaleAspect(),
        T.RandomBlur(),
        T.Resize(target_size=(512, 512)),
        T.Normalize()
    ]
    train_dataset = Dataset(
        transforms=train_transforms,
        dataset_root=dataset_root,
        num_classes=classes,
        mode="train",
        train_path=osp.join(dataset_root, "train.txt"),
        separator=" "
    )

    val_transforms = [
        InitMask(),
        T.Resize(target_size=(512, 512)),
        T.Normalize()
    ]
    val_dataset = Dataset(
        transforms=val_transforms,
        dataset_root=dataset_root,
        num_classes=classes,
        mode="train",
        train_path=osp.join(dataset_root, "val.txt"),
        separator=" "
    )
    
    return train_dataset, val_dataset