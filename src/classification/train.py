
import torch.optim as optim
import torch
from model import resnet
from datasets import MyDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

nhparams = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "data_dir":"Datasets/scannet/scans/nbv_data.txt",
    "num_classes": 4,
    "num_workers": 4,    # used by the dataloader, more workers means faster data preparation, but for us this is not a bottleneck here
}

model = resnet(hparams=nhparams)
logger = TensorBoardLogger('logs', name='depth and rgb_8000')

trainer = pl.Trainer(
    enable_model_summary=True,
    max_epochs=100,
    gpus=1,
    logger=logger,
    log_every_n_steps=2,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=30)]
)


trainer.fit(model)



# model = resnet.load_from_checkpoint("logs/depth with mask/version_2/checkpoints/epoch=75-step=7220.ckpt")
# model.getTest()



# 13: npy: rgb+depth 4 layers
# 17: png: depth 3 layers
# version 9: only depth png
# version 13: depth and rgb pics add together


# version 2: depth and rgb (0-1) 4 layers
