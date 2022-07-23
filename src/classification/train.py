
import torch.optim as optim
import torch
from model import alexnet
from datasets import MyDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

nhparams = {
    "batch_size": 4,
    "learning_rate": 1e-3,
    "data_dir":"Datasets/scannet/scans/nbv_data.txt",
    # "input_size": 1 * 28 * 28,
    # "hidden_size": 512,
    "num_classes": 4,
    "num_workers": 4,    # used by the dataloader, more workers means faster data preparation, but for us this is not a bottleneck here
}
model = alexnet(hparams=nhparams)
logger = TensorBoardLogger('logs', name='model')

trainer = pl.Trainer(
    enable_model_summary=True,
    max_epochs=20,
    gpus=1,
    logger=logger,
    log_every_n_steps=2
)


trainer.fit(model)

a,b = model.getTestAcc()
print(b)
