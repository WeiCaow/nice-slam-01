import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import MyDataset
from torch.utils.data import DataLoader, random_split
from torchvision import models  
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

class alexnet(pl.LightningModule):

    def __init__(self, hparams):
      super().__init__()

      self.save_hyperparameters(hparams)

      self.model  = models.resnet18(pretrained=True)
      self.model.conv1= nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      self.model.fc = nn.Linear(512, self.hparams["num_classes"])
 

      dataset = MyDataset(self.hparams["data_dir"])
      train_size = int(len(dataset) * 0.8)
      val_size = int(len(dataset) * 0.1)
      test_size = len(dataset) - train_size - val_size
      self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])


    def forward(self, x):

      x = self.model(x)
      return x

    def training_step(self, batch, batch_idx):

      images, targets = batch

      # Perform a forward pass on the network with inputs
      out = self.forward(images)

      # calculate the loss with the network predictions and ground truth targets
      loss = F.cross_entropy(out, targets)

      # Find the predicted class from probabilities of the image belonging to each of the classes
      # from the network output
      _, preds = torch.max(out, 1)

      # Calculate the accuracy of predictions
      acc = preds.eq(targets).sum().float() / targets.size(0)

      # Log the accuracy and loss values to the tensorboard
      self.log('loss', loss)
      self.log('acc', acc)


      return {'loss': loss,'acc': acc}

    def validation_step(self, batch, batch_idx):
      images, targets = batch

      # Perform a forward pass on the network with inputs
      out = self.forward(images)

      # calculate the loss with the network predictions and ground truth targets
      loss = F.cross_entropy(out, targets)

      # Find the predicted class from probabilities of the image belonging to each of the classes
      # from the network output
      _, preds = torch.max(out, 1)

      # Calculate the accuracy of predictions
      acc = preds.eq(targets).sum().float() / targets.size(0)

      return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):

      # Average the loss over the entire validation data from it's mini-batches
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

      # Log the validation accuracy and loss values to the tensorboard
      self.log('val_loss', avg_loss)
      self.log('val_acc', avg_acc)

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])

        return optim

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams["batch_size"],shuffle = True,num_workers=4,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"],shuffle = False,num_workers=4,drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams["batch_size"])

    def getTestAcc(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


if __name__ == "__main__":
  import torchvision.models as models
  backbone = models.resnet18(pretrained=False)
  backbone.conv1= nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)