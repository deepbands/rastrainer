import os
import os.path as osp
import paddle
from paddleseg.models import OCRNet, HRNet_W18
from paddleseg.models.segformer import SegFormer_B2
from paddleseg.models.bisenet import BiSeNetV2
from paddleseg.models.losses import MixedLoss, CrossEntropyLoss, DiceLoss
from paddleseg.core import train


MODELS = ["OCRNet_HRNetw18", "SegFormer_B2", "BiSeNetV2"]


class Model:
    def __init__(self, model_name, classes=2, pretrained=None) -> None:
        if model_name not in MODELS:
            raise ValueError(f"model_name must be in {MODELS}, but now is {model_name}")
        if model_name == "OCRNet_HRNetw18":
            self.net = OCRNet(num_classes=classes, backbone=HRNet_W18(), backbone_indices=[0])
            self.loss_coef = [1, 0.4]
        elif model_name == "SegFormer_B2":
            self.net = SegFormer_B2(num_classes=classes)  # BUG: load crash
            self.loss_coef = [1]
        else:  # model_name == "BiSeNetV2"
            self.net = BiSeNetV2(num_classes=classes)
            self.loss_coef = [1, 1, 1, 1, 1]
        if pretrained is not None:
            self.load_weight(pretrained)
        print(f"init {model_name} model successfully.")

    def load_weight(self, pretrained):
        if osp.exists(pretrained) and pretrained.split(".")[-1] == "pdparams":
            pre_weight = paddle.load(pretrained)
            self.net.set_state_dict(pre_weight)
        print("load weight finished.")

    def train(self, args_dict):
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
        # loss
        lr = paddle.optimizer.lr.LinearWarmup(base_lr, 
                                              warmup_steps=iters // epochs, 
                                              start_lr=base_lr / 10, 
                                              end_lr=base_lr)
        # lr = paddle.optimizer.lr.PolynomialDecay(base_lr, 
        #                                          decay_steps=iters // epochs, 
        #                                          end_lr=base_lr / 5)
        optimizer = paddle.optimizer.AdamW(lr, parameters=self.net.parameters())
        losses = {}
        losses["types"] = [MixedLoss([CrossEntropyLoss(), DiceLoss()], [1, 1])] * len(self.loss_coef)
        losses["coef"] = self.loss_coef
        # train
        train(model=self.net,
              train_dataset=train_dataset,
              val_dataset=val_dataset,
              optimizer=optimizer,
              save_dir=output_dir,
              iters=iters,
              batch_size=batch_size,
              save_interval=iters // save_number,
              log_iters=log_iters,
              num_workers=0,
              losses=losses,
              use_vdl=True)


if __name__ == "__main__":
    model = Model("OCRNet_HRNetw18")
    print(model)