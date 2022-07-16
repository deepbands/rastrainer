import os
import os.path as osp
import paddle
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer import AdamW
from paddleseg.models import OCRNet, HRNet_W18
from paddleseg.models.segformer import SegFormer_B2
from paddleseg.models.bisenet import BiSeNetV2
from paddleseg.models.unet import UNet
from paddleseg.models.losses import MixedLoss, CrossEntropyLoss, DiceLoss
from paddleseg.core import train
from typing import Dict, Optional


MODELS = ["OCRNet_HRNetw18", "SegFormer_B2", "BiSeNetV2", "UNet"]


class Model:
    def __init__(self, 
                 model_name: str, 
                 classes: int=2, 
                 pretrained: Optional[str]=None) -> None:
        self.init_model(model_name, classes, pretrained)

    def init_model(self, 
                   model_name: str, 
                   classes: int, 
                   pretrained: Optional[str]) -> None:
        if model_name not in MODELS:
            raise ValueError(f"model_name must be in {MODELS}, but now is {model_name}")
        if model_name == "OCRNet_HRNetw18":
            self.net = OCRNet(
                num_classes=classes, 
                backbone=HRNet_W18(), 
                backbone_indices=[0],
                pretrained="https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams"
            )
            self.loss_coef = [1, 0.4]
        elif model_name == "SegFormer_B2":
            self.net = SegFormer_B2(
                num_classes=classes,
                pretrained="https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b2_cityscapes_1024x1024_160k/model.pdparams"
            )
            self.loss_coef = [1]
        elif model_name == "BiSeNetV2":
            self.net = BiSeNetV2(
                num_classes=classes,
                pretrained="https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenetv1_resnet18_os8_cityscapes_1024x512_160k/model.pdparams"
            )
            self.loss_coef = [1, 1, 1, 1, 1]
        else:  # model_name == "UNet":
            self.net = UNet(
                num_classes=classes,
                pretrained="https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/model.pdparams"
            )
            self.loss_coef = [1]
        if pretrained is not None:
            self._load_weight(pretrained)
        else:
            print("load default weight based on cityscapes finished.")
        print(f"init {model_name} model successfully.")

    def _load_weight(self, pretrained: str) -> None:
        if osp.exists(pretrained) and pretrained.split(".")[-1] == "pdparams":
            pre_weight = paddle.load(pretrained)
            self.net.set_state_dict(pre_weight)
        print("load user's weight finished.")

    def _default_setting(self, base_lr: float, iters: int, epochs: int) -> None:
        # default learning rate
        lr = LinearWarmup(
            base_lr, 
            warmup_steps=iters // epochs, 
            start_lr=base_lr / 10, 
            end_lr=base_lr
        )
        # default optimizer
        self.optimizer = AdamW(lr, parameters=self.net.parameters())
        # default loss
        self.losses = {}
        self.losses["types"] = [
            MixedLoss([CrossEntropyLoss(), DiceLoss()], [1, 1])] * len(self.loss_coef)
        self.losses["coef"] = self.loss_coef

    def train(self, args_dict: Dict) -> None:
        # setting
        base_lr = args_dict["learning_rate"]
        epochs = args_dict["epochs"]
        batch_size = args_dict["batch_size"]
        train_dataset = args_dict["train_dataset"]
        val_dataset = args_dict["val_dataset"]
        output_dir = args_dict["save_dir"]
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        save_number = args_dict["save_number"]
        log_iters = args_dict["log_iters"]
        iters = epochs * len(train_dataset) // batch_size
        self._default_setting(base_lr, iters, epochs)
        # train
        train(
            model=self.net,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=self.optimizer,
            save_dir=output_dir,
            iters=iters,
            batch_size=batch_size,
            save_interval=iters // save_number,
            log_iters=log_iters,
            num_workers=0,
            losses=self.losses,
            use_vdl=True
        )
