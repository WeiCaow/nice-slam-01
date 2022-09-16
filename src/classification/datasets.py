from email.mime import image
from hashlib import algorithms_available
from statistics import median_grouped
from timeit import repeat
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
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from skimage.transform import rescale, resize

input_folder = "Datasets/scannet/scans/scene0000_00"

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None, pad=False):

        fh = open(datatxt, 'r')
        imgs = []
        labels = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            labels.append(int(words[1]))
        self.imgs = imgs
        self.pad = pad
        self.labels = labels

    def __getitem__(self, index):

        dir, label = self.imgs[index]
        if self.pad is True:
            candidate_list = ["y+degree","y-degree","x+degree","x-degree"]
            result_list = []
             # convert rgb image to grayscale
            img = cv2.imread(os.path.join(dir,"rgb_big.jpg"))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            depth = resize(gray,(400,400))

            for i,ca in enumerate(candidate_list):
                result_mtx = np.zeros_like(depth)
                tri_mtx = np.tri(400,400)
                tri_mtx_rot = np.rot90(tri_mtx)
                mask_mtx = np.logical_and(tri_mtx_rot, tri_mtx)
                if ca == "y-degree":
                    result_mtx[mask_mtx] = depth[mask_mtx]
                elif ca == "x-degree":
                    mask_mtx = np.rot90(mask_mtx,1)
                    result_mtx[mask_mtx] = depth[mask_mtx]
                elif ca == "y+degree":
                    mask_mtx = np.rot90(mask_mtx,2)
                    result_mtx[mask_mtx] = depth[mask_mtx]
                elif ca == "x+degree":
                    mask_mtx = np.rot90(mask_mtx,3)
                    result_mtx[mask_mtx] = depth[mask_mtx]
                result_list.append(torch.from_numpy(result_mtx).unsqueeze(-1).float())
            
            img = torch.cat(result_list,-1)
            # img = torch.cat([torch.from_numpy(depth).unsqueeze(-1),img],-1)
        else:
            # depth = torch.from_numpy(np.load(os.path.join(dir,"depth_big.npy")))
            # depth = np.clip(depth,0,10)/depth.max()
            # depth = depth/depth.max()
            rgb = torch.from_numpy(np.load(os.path.join(dir,"rgb_big.npy"))).float()
            # img = torch.from_numpy(plt.imread(os.path.join(dir,"depth_big.png")))
            # rgb =  torch.from_numpy(plt.imread((os.path.join(dir,"rgb_big.jpg")))).float()
            # img  = torch.cat([depth.unsqueeze(-1),rgb],-1).float()
            # img = depth.unsqueeze(-1).repeat(1,1,3)
            

            # img = depth
        
        
            

            # img = torch.from_numpy(cv2.imread(os.path.join(dir,"depth_big.png"))).float()

        img = rgb.permute(2,0,1)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_labels(self): return self.labels

