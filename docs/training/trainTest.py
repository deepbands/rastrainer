#%% 5 Package import（包导入）
import paddle
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
# from paddleseg.models import OCRNet, HRNet_W18  # OCRNet
from paddleseg.models.segformer import SegFormer_B2  # segformer
from paddleseg.models.losses import MixedLoss, BCELoss, DiceLoss, LovaszHingeLoss
from paddleseg.core import train, evaluate


#%% * Label processing（标签处理）
import numpy as np
from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
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


#%% 6 Data set establishment（数据集建立）
# Build the training set
# 建立训练数据集
train_transforms = [
    InitMask(),  # This is not necessary（这不是必要的）
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
    dataset_root="data/dataset",
    num_classes=2,
    mode="train",
    train_path="data/dataset/train.txt",
    separator=" "
)

# Build validation set
# 建立评估数据集
val_transforms = [
    InitMask(),
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]
val_dataset = Dataset(
    transforms=val_transforms,
    dataset_root="data/dataset",
    num_classes=2,
    mode="train",  # If your data does not need `initmask()`, please use "val"（如果你的数据不需要`InitMask()`，请使用val"）
    train_path="data/dataset/val.txt",
    separator=" "
)


#%% * Check dataset（检查数据）
# for img, lab in val_dataset:
#     print(img.shape, lab.shape)
#     print(np.unique(lab))


#%% 7 Training parameter setting（训练参数设置）
base_lr = 3e-5
epochs = 2
batch_size = 16
iters = epochs * len(train_dataset) // batch_size

# model = OCRNet(num_classes=2,
#                backbone=HRNet_W18(),
#                backbone_indices=[0],
#                pretrained="output/best_model/model.pdparams")

model = SegFormer_B2(num_classes=2,
                     pretrained="output_segformer/best_model/model.pdparams")

# lr = paddle.optimizer.lr.LinearWarmup(base_lr, warmup_steps=iters // epochs, start_lr=base_lr / 10, end_lr=base_lr)
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, decay_steps=iters // epochs, end_lr=base_lr / 5)
optimizer = paddle.optimizer.AdamW(lr, beta1=0.9, beta2=0.999, weight_decay=0.01, parameters=model.parameters())
losses = {}
losses["types"] = [MixedLoss([BCELoss(), DiceLoss(), LovaszHingeLoss()], [2, 1, 1])]  #  * 2
losses["coef"] = [1]  # [1, 0.4]


#%% 8 Start training（开始训练）
model.train()

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir="output_segformer",
    iters=iters,
    batch_size=batch_size,
    save_interval=iters // 5,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)

#%% 9 Model evaluation（模型评估）
# model.eval()

# evaluate(
#     model,
#     val_dataset)