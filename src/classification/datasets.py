from email.mime import image
from hashlib import algorithms_available
from statistics import median_grouped
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
image_dir = "Datasets/scannet/scans"




dirs = sorted(os.listdir(image_dir))
for dir in dirs:
    if not os.path.isdir(os.path.join(image_dir,dir)):
        dirs.remove(dir)

f = open("Datasets/scannet/scans/nbv_data.txt", "w")
for dir in dirs[:100]:
    render_dir = os.path.join(image_dir, dir,"render")
    frames = sorted(os.listdir(render_dir))
    for frame in frames:
        big_FOV = os.path.join(render_dir,frame)
        nbv_dir = os.path.join(image_dir,dir,"nbv",f"{frame}.txt")
        nbv = np.loadtxt(nbv_dir)
        nbv = np.argmax(nbv)
        f.write(str(big_FOV)+" "+str(nbv)+'\n')
f.close()





class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None, pad=False):

        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.pad = pad

    def __getitem__(self, index):

        dir, label = self.imgs[index]
        if self.pad is True:
            candidate_list = ["y+degree","y-degree","x+degree","x-degree"]
            result_list = []
            depth = plt.imread(os.path.join(dir,"depth_big.png"))
            depth = resize(depth,(400,400))
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
                result_list.append(torch.from_numpy(result_mtx).float())
            
            img = torch.cat(result_list,-1)
            # img = torch.cat([torch.from_numpy(depth).unsqueeze(-1),img],-1)

            





        else:
            depth = torch.from_numpy(np.load(os.path.join(dir,"depth_big.npy")))
            depth = np.clip(depth,0,10)/depth.max()
            rgb =  torch.from_numpy(np.load(os.path.join(dir,"rgb_big.npy"))/255)
            # img = torch.from_numpy(plt.imread(os.path.join(dir,"depth_big.png")))
            # rgb =  torch.from_numpy(plt.imread((os.path.join(dir,"rgb_big.jpg")))).float()
            img  = torch.cat([depth.unsqueeze(-1),rgb],-1).float()

            # H,W = img.shape[0], img.shape[1]
            # fy = 578.729797
            # fx = 577.590698

            # H = int(np.tan(np.deg2rad(np.rad2deg(np.arctan(H/(2*fy)))+10))*fy*2)
            # W = int(np.tan(np.deg2rad(np.rad2deg(np.arctan(W/(2*fx)))+10))*fx*2)

            

            # img = torch.from_numpy(cv2.imread(os.path.join(dir,"depth_big.png"))).float()

        img = img.permute(2,0,1)
        return img, label

    def __len__(self):
        return len(self.imgs)