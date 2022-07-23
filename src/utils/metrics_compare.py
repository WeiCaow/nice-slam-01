import os
import glob
import pandas as pd
import numpy as np
nbv_dir = "Datasets/scannet/scans/scene0000_00/nbv"
metrics_dir = "Datasets/scannet/scans/scene0000_00/render"

def compare(nbv_dir,metrics_dir):
  metrics_path = []
  for root, dirs, files in os.walk(metrics_dir):
    for dir in dirs:
       metrics_path.append(os.path.join(metrics_dir,dir, f'{dir}.csv'))
  metrics_path = sorted(metrics_path,
                        key=lambda x: int(os.path.basename(x)[:-4]))
  top_1 = []
  top_2 = []
  for i,metric in enumerate(metrics_path):

    # first value: index of the samllest value
    nbv = np.argsort(np.loadtxt(os.path.join(nbv_dir,f'{os.path.basename(metric[:-4])}.txt')))
    metric_order = np.argsort(pd.read_csv(metric, index_col=0).to_numpy(), axis=0)
    nbv = nbv[:,None].repeat(metric_order.shape[-1], axis=1)
    top_1.append(nbv[0,:]==metric_order[0,:])

    # top_20 = np.logical_or.reduce(nbv[:2,:]==metric_order[:2,:],axis = 0)
    # top_21 = np.logical_or.reduce(nbv[:2,:][[1,0],:]==metric_order[:2,:],axis = 0)
    top_2.append((nbv[0,:]== metric_order[0,:])|(nbv[0,:]== metric_order[1,:]))
    if i >  100 :
      break
  
  top_1 = np.mat(top_1)
  top_2 = np.mat(top_2)
  top1 = top_1.sum(axis=0)/top_1.shape[0]
  top2 = top_2.sum(axis=0)/top_2.shape[0]

  return top1,top2


top1,top2 = compare(nbv_dir,metrics_dir)
print(top1.max(),top2.max())
