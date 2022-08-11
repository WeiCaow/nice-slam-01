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

def gt_nbv_generation (input_folder, move=10, freq=20):
    "[orgin, 4 different candidates]"
    mesh_dir = glob.glob(os.path.join(input_folder,"*.ply"))[0]
    mesh = o3d.io.read_triangle_mesh(mesh_dir)
    pose_folder = os.path.join(input_folder, 'pose')
    gt_folder = os.path.join(input_folder, 'nbv')
    pose_paths = sorted(glob.glob(os.path.join(pose_folder, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    pose_list = []
    for pose_path in pose_paths:
        with open(pose_path, "r") as f:
            lines = f.readlines()
        ls = []
        for line in lines:
            l = list(map(float, line.split(' ')))
            ls.append(l)
        c2w = np.array(ls).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        pose_list.append(c2w)

    os.makedirs(gt_folder, exist_ok=True)

    for idx, pose in enumerate(pose_list):
        if idx%freq == 0:
            print(input_folder,idx)
            candidate_generate_list = candidate_generate_np(pose, move=move)
            if idx == 0:
                num_pcd, pcd = headless_pcd_render(mesh, [pose_list[idx]])
            else:
                num_pcd, pcd = headless_add_one_pcd_render(mesh, pose=pose_list[idx], p=pcd)
            pcd_ca_num = headless_pcd_ca_render(mesh,candidate_generate_list,pcd=pcd)
            np.savetxt(os.path.join(gt_folder, f"{idx:05d}.txt"), pcd_ca_num)
            
    return 0

# if __name__ == '__main__':

input_folder = "Datasets/scannet/scans"
for root, dirs, files in os.walk(input_folder):
    dir = sorted(dirs)
    for name in dir[125:126]:
        # print(name)
        gt_nbv_generation(os.path.join(root,name))

