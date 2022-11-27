
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from candidate_renderer import *
import open3d as o3d
import cv2

nbv_dir = "Datasets/scannet/scans/scene0000_00/nbv"
metrics_dir = "Datasets/scannet/scans/scene0000_00/render"
scans_dir = "Datasets/scannet/scans"

def compare(nbv_dir,metrics_dir,scans_dir):

  top_1 = []
  top_2 = []
  dirs = sorted(os.listdir(scans_dir))
  for dir in dirs[4:]:
    render_dir = os.path.join(scans_dir,dir,"render")
    frames = sorted(os.listdir(render_dir))
    for frame in frames:
      frame_dir = os.path.join(render_dir,frame,f"{frame}.csv")
      # metric 越小越好 0-smallest 3-biggest
      metric_order = np.argsort(pd.read_csv(frame_dir, index_col=0).to_numpy(), axis=0)
      nbv_dir = os.path.join(scans_dir,dir,"nbv",f"{frame}.txt")
      nbv = np.loadtxt(nbv_dir)
      nbv = np.argsort(nbv)


      nbv = nbv[:,None].repeat(metric_order.shape[-1], axis=1)
      top_1.append(nbv[0,:]==metric_order[0,:])

      top_2.append((nbv[0,:]== metric_order[0,:])|(nbv[0,:]== metric_order[1,:]))

          
  top_1 = np.mat(top_1)
  top_2 = np.mat(top_2)
  top1 = top_1.sum(axis=0)/top_1.shape[0]
  top2 = top_2.sum(axis=0)/top_2.shape[0]










  # metrics_path = []
  # for root, dirs, files in os.walk(metrics_dir):
  #   for dir in dirs:
  #      metrics_path.append(os.path.join(metrics_dir,dir, f'{dir}.csv'))
  # metrics_path = sorted(metrics_path,
  #                       key=lambda x: int(os.path.basename(x)[:-4]))
  # top_1 = []
  # top_2 = []
  # for i,metric in enumerate(metrics_path):

  #   # first value: index of the samllest value
  #   nbv = np.argsort(np.loadtxt(os.path.join(nbv_dir,f'{os.path.basename(metric[:-4])}.txt')))[::-1]
  #   metric_order = np.argsort(pd.read_csv(metric, index_col=0).to_numpy(), axis=0)
  #   nbv = nbv[:,None].repeat(metric_order.shape[-1], axis=1)
  #   top_1.append(nbv[0,:]==metric_order[0,:])

  #   # top_20 = np.logical_or.reduce(nbv[:2,:]==metric_order[:2,:],axis = 0)
  #   # top_21 = np.logical_or.reduce(nbv[:2,:][[1,0],:]==metric_order[:2,:],axis = 0)
  #   top_2.append((nbv[0,:]== metric_order[0,:])|(nbv[0,:]== metric_order[1,:]))
  #   if i >  800 :
  #     break
  
  # top_1 = np.mat(top_1)
  # top_2 = np.mat(top_2)
  # top1 = top_1.sum(axis=0)/top_1.shape[0]
  # top2 = top_2.sum(axis=0)/top_2.shape[0]

  return top1,top2


top1,top2 = compare(nbv_dir,metrics_dir, scans_dir)
print(top1.max(),top2.max())
print(top1,top2)


