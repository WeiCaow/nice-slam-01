import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import MyDataset
from torch.utils.data import DataLoader, random_split
from torchsampler import ImbalancedDatasetSampler
from torchvision import models  
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
import torchmetrics
from vit_pytorch import ViT

def make_weights_for_balanced_classes(images, nclasses):                        
        count = [0] * nclasses                                                      
        for item in images:                                                         
            count[item[1]] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[1]]                                  
        return torch.DoubleTensor(weight)  


class resnet(pl.LightningModule):

    def __init__(self, hparams):
      super().__init__()

      self.save_hyperparameters(hparams)
      if self.hparams["model"] == "resnet":
        self.model = models.resnet18()
        self.model.conv1= nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Sequential(
          nn.Linear(512, 32),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          # nn.Linear(128, 32),
          # nn.ReLU(),
          # nn.Dropout(p=0.2),
          nn.Linear(32, self.hparams["num_classes"])
        )
      elif self.hparams["model"] == "ViT":
        self.model = ViT(
            image_size = (460, 620),
            patch_size = 20,
            num_classes = 4,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )


      self.train_dataset = MyDataset(self.hparams["data_dir"]+"nbv_data_train.txt")
      self.val_dataset = MyDataset(self.hparams["data_dir"]+"nbv_data_val.txt")
      self.test_dataset = MyDataset(self.hparams["data_dir"]+"nbv_data_test.txt")

      # self.balanced_weights = dataset.balanced_weights()
      # self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.balanced_weights, len(self.balanced_weights))

      # self.balanced_weights = make_weights_for_balanced_classes(self.train_dataset, 4)
      # self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.balanced_weights, len(self.balanced_weights))   
      # Initialize the Weight Transforms
      # weights = models.ResNet18_Weights.DEFAULT
      # self.preprocess = weights.transforms()

      

    def forward(self, x):

        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        # images = self.preprocess(images)
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Log the accuracy and loss values to the tensorboard
        self.log('loss', loss)
        
        # Or also on the progress bar in our console/notebook
        # if you want it to not show in tensorboard just disable
        # the logger but usually you want to log everything :)
        self.log('acc', acc, logger=True, prog_bar=True)

        # Ultimately we return the loss which will be then used
        # for backpropagation in pytorch lightning automatically
        # This will always be logged in the progressbar as well
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        # images = self.preprocess(images)

        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Whatever we return here, we have access to in the 
        # validation epoch end function. A dictionary is more
        # ordered than a list or tuple
        self.log("val_loss", loss, logger=True, prog_bar=True)
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
        return DataLoader(self.train_dataset, batch_size=self.hparams["batch_size"], num_workers=4, sampler=ImbalancedDatasetSampler(self.train_dataset), drop_last=True)
        # sampler=ImbalancedDatasetSampler(self.train_dataset),

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=4, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams["batch_size"])

    def getTest(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            # X = self.preprocess(X)
            
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        # acc = (labels == preds).mean()
        print(classification_report(labels,preds))
        return 0


if __name__ == "__main__":
  import torchvision.models as models
  backbone = models.resnet18(pretrained=False)
  backbone.conv1= nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)