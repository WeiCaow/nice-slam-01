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
input_folder = "Datasets/scannet/scans/scene0000_00"
image_folder = "output/scannet/scans/scene0000_00/metrics"
gt_folder = os.path.join(input_folder, 'frames', 'nbv')

nbv_dir = sorted(glob.glob(os.path.join(gt_folder, '*.txt')),
                    key=lambda x: int(os.path.basename(x)[:-4]))


rgb_images = [
          "render_rgb_x+Degree.jpg",
          "render_rgb_x-Degree.jpg",
          "render_rgb_y+Degree.jpg",
          "render_rgb_y-Degree.jpg"
          ]

depth_images = [
          "render_depth_x+Degree.png",
          "render_depth_x-Degree.png",
          "render_depth_y+Degree.png",
          "render_depth_y-Degree.png"
]




rgb_img = []
depth_img = []

label = []
for root, dirs, files in os.walk(image_folder):
    dirs = sorted(dirs)
    for name in dirs:
      init = np.zeros(4).astype('int')
      metric = np.argmax(np.loadtxt(os.path.join(gt_folder,f'{name}.txt')))
      init[metric] = 1
      label.extend(init.tolist())
      for i in range(4):
        rgb_img.append(os.path.join(root, name, rgb_images[i]))
        depth_img.append(os.path.join(root, name, depth_images[i]))
      if len(rgb_img)>(len(nbv_dir)/5 - 2)*4:
        break
      
 
f=open("Datasets/scannet/scans/scene0000_00/nbv_data_train.txt","w")
g=open("Datasets/scannet/scans/scene0000_00/nbv_data_val.txt","w")
k=open("Datasets/scannet/scans/scene0000_00/nbv_data_test.txt","w")
for i in range(len(rgb_img)) :
    if i < len(rgb_img)//10*8:
      f.write(str(rgb_img[i])+' '+str(depth_img[i])+' '+str(label[i])+'\n')
    elif i< len(rgb_img)//10*9:
      g.write(str(rgb_img[i])+' '+str(depth_img[i])+' '+str(label[i])+'\n')
    elif i< len(rgb_img):
      k.write(str(rgb_img[i])+' '+str(depth_img[i])+' '+str(label[i])+'\n')

f.close()
g.close()
k.close()



 
class MyDataset(torch.utils.data.Dataset): 
    def __init__(self,datatxt, transform=None, target_transform=None): 
        
        fh = open(datatxt, 'r') 
        imgs = [] 
        for line in fh:  
            line = line.rstrip()   
            words = line.split()   
            imgs.append((words[0],words[1],int(words[2]))) 
                         
        self.imgs = imgs
    
 
    def __getitem__(self, index): 
        rgb, depth, label = self.imgs[index]
        rgb = cv2.imread(rgb)
        depth = cv2.imread(depth,-1)
        t = transforms.ToTensor()
        rgb = t(rgb)
        depth = t(depth)
        img = torch.cat((rgb,depth),0)

        return img,label
 
    def __len__(self): 
        return len(self.imgs)

 

# root = "Datasets/scannet/scans/scene0000_00/nbv_data.txt"
# train_data=MyDataset(root)
# test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())


 






# save_root = '/media/1TB Hard Disk'
# data_filename = '{}/grasp_{}_center.npy'.format(save_root, split)
# data_filename_gt = '{}/grasp_{}_m.npy'.format(save_root, split)
# # label_filename = '{}/grasp_{}_label_new.npy'.format(save_root, split)
# data = open_memmap(data_filename, dtype='float32', mode='w+', shape=(len(filepaths), 2048, 3))
# data_gt = open_memmap(data_filename_gt, dtype='float32', mode='w+', shape=(len(filepaths), 8192, 3))
# # label = open_memmap(label_filename, dtype='float32', mode='w+', shape=(len(filepaths)))
# for idx in tqdm(range(len(filepaths)), desc='{} samples'.format(split)):
#     path = filepaths[idx]
#     class_name = path.split('/')[-3]
#     class_id = self.classnames.index(class_name)
#     partial = np.loadtxt(filepaths[idx])
#     # gt_dir = pcd_dir + '/' + class_name + '/{}/'.format(split) + os.path.split(str(path))[-1][:-4] + '.xyz'
#     # p =






# for time in range(1000):
#   volume.integrate(rgbd_ca,camera_intrinsics,timestep[i])
#   for i in range(len(test)):
#       volume.integrate(rgbd_ca,camera_intrinsics,test[i])
#       print("point cloud number of pose 0[volume]", np.asarray(volume.extract_point_cloud().points).shape[0])
#       o3d.io.write_triangle_mesh('mesh{}.ply'.format(i),volume.extract_triangle_mesh())
#       volume.reset()