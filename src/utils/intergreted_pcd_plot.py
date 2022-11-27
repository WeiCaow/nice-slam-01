# plot the intergrated nbv point cloud

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import os
import glob
import matplotlib.pyplot as plt
import copy
import cv2
import torch
from candidate_renderer import headless_render, headless_render_cadidate, candidate_generate, headless_add_one_render, headless_pcd_ca_render, headless_pcd_render, headless_add_one_pcd_render,candidate_generate_np

mesh_dir = "/home/cao/Desktop/nice-slam-01/Datasets/Replica/office2_mesh.ply"
mesh = o3d.io.read_point_cloud(mesh_dir)

output_dir = "/home/cao/Desktop/nice-slam-01/output/Replica/office2/NBV_render/NRSS_RGB"
pose_dir = "/home/cao/Desktop/nice-slam-01/output/Replica/office2/NBV_render/NRSS_RGB"

dirs = sorted(os.listdir(pose_dir))

pose_list = []
for dir in dirs[:-2]:
  a = np.loadtxt(glob.glob(os.path.join(pose_dir,dir,"*.txt"))[0])
  a[:3, 1] *= -1
  a[:3, 2] *= -1
  pose_list.append(a)
  # pose_list.append(np.loadtxt(os.path.join(pose_dir,dir)))

pcd_num, pcd = headless_pcd_render(mesh,pose_list[:50])


plt.plot(range(len(pcd_num)),pcd_num)
plt.savefig(os.path.join(output_dir,"pcd.jpg"))
o3d.io.write_point_cloud(os.path.join(output_dir,"pcd.ply"), pcd)

print(pcd_num)