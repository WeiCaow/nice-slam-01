
import torch.optim as optim
import torch
from model import resnet
from datasets import MyDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

nhparams = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "data_dir":"Datasets/scannet/scans/",
    "num_classes": 4,
    "num_workers": 4,
    "model": "ViT"    # used by the dataloader, more workers means faster data preparation, but for us this is not a bottleneck here
}

model = resnet(hparams=nhparams)
logger = TensorBoardLogger('logs', name='ViT_20000_resnet18')

trainer = pl.Trainer(
    enable_model_summary=True,
    max_epochs=100,
    gpus=1,
    logger=logger,
    log_every_n_steps=2,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=20)]
)


trainer.fit(model)



# model = resnet.load_from_checkpoint("logs/ragb_gray_mask_20000_resnet18/version_2/checkpoints/epoch=35-step=9720.ckpt")
# model.getTest()



# new_rgb_20000_pretrained
    # Version 8: resnet50 pretrained

# depthx3_20000_pretrained
    # Version 4: resnent50 pretrained
    # Version 6: resnet152 pretrained


# 21 sample
# 23 without sample
# 26 split 直接掉包sample
# 27 split 自己写的方法sample
# 30 brute force sample