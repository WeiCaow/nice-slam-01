from email.mime import image
from hashlib import algorithms_available
from tkinter import Frame
import numpy as np
import os
import torch
import glob
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import open3d as o3d
import matplotlib.pyplot as plt
import json
input_folder = "Datasets/scannet/scans/scene0000_00"

gt_folder = os.path.join(input_folder, 'nbv')
render_folder = os.path.join(input_folder, 'render')

nbv_dir = sorted(glob.glob(os.path.join(gt_folder, '*.txt')),
                 key=lambda x: int(os.path.basename(x)[:-4]))

f = open("Datasets/scannet/scans/nbv_data.txt", "w")
for root, dirs, files in os.walk(render_folder):
    dirs = sorted(dirs)
    for name in dirs:
        nbv = np.argmax(np.loadtxt(os.path.join(gt_folder, f'{name}.txt')))
        f.write(str(os.path.join(render_folder, f'{name}'))+" "+str(nbv)+'\n')
        if name == "03010":
            break
f.close()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):

        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs

    def __getitem__(self, index):
        dir, label = self.imgs[index]
        depth = torch.from_numpy(np.load(os.path.join(dir,"render_depth_large_scale.npy")))
        rgb =  torch.from_numpy(np.load(os.path.join(dir,"render_rgb_large_scale.npy")))
        
        img  = torch.cat([depth.unsqueeze(-1),rgb],-1).float()
        img = img.permute(2,0,1)
        return img, label

    def __len__(self):
        return len(self.imgs)