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
from candidate_renderer import headless_render, headless_render_cadidate, candidate_generate, headless_add_one_render, headless_pcd_ca_render, headless_pcd_render, headless_add_one_pcd_render

input_folder = "Datasets/scannet/scans/scene0000_00"


def gt_nbv_generation (input_folder, move=10, use_pcd = True):
    "[orgin, 4 different candidates]"
    mesh = o3d.io.read_triangle_mesh(
        "/home/cao/Desktop/nice-slam-01/Datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply")
    pose_folder = os.path.join(input_folder, 'frames', 'pose')
    gt_folder = os.path.join(input_folder, 'frames', 'nbv')
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



    # pcd_num, volume = headless_render(mesh,pose_list[:1])
    for idx, pose in enumerate(pose_list):
        candidate_generate_list = candidate_generate(pose.cuda(), move=move)
        if use_pcd:
            if idx == 0:
                num_pcd, pcd = headless_pcd_render(mesh, pose_list[:1])
            else:
                num_pcd, pcd = headless_add_one_pcd_render(mesh, pose=pose, p=pcd)
            pcd_ca_num = headless_pcd_ca_render(mesh,candidate_generate_list,pcd=pcd)
            np.savetxt(os.path.join(gt_folder, f"{idx:05d}.txt"), pcd_ca_num)

        # while idx>1:
        #     volume_ca = copy.deepcopy(volume)
        #     num_pcd, volume_ca = headless_add_one_render(mesh,pose,v=volume_ca)

            

                # num_pcd,_,_ = headless_render_cadidate(mesh,candidate_generate_list, history=True, v=volume)
                # np.savetxt(os.path.join(gt_folder, f"{idx:05d}_volu.txt"), num_pcd)

        

#c,d = headless_pcd_render(mesh, pose_list[:8])


    # num_1, v_1 = headless_render(mesh, pose_list[:3])
    #
    # num_0, v_0 = headless_render(mesh, pose_list[:1])
    # v_0, num_2 = headless_add_one_render(mesh, pose=pose_list[1], v=v_0)
    # v_0, num_3 = headless_add_one_render(mesh, pose=pose_list[2], v=v_0)
    # num_pcd_0, volume = headless_render(mesh, pose_list[:1])
    # num_pcd, volume = headless_add_one_render(mesh, pose=pose_list[1], v=volume)
    # global volu
    # v_list = []
    # for idx, pose in enumerate(pose_list):
    #     if idx == 0:
    #         num_pcd_0, volu = headless_render(mesh, pose_list[:1])
    #         v_list.append(volu)
    #     else:
    #         num_pcd, volu = headless_add_one_render(mesh, pose=pose, v=volume)
    #         v_list.append(volu)
    #
    #
    #
    #     candidate_generate_list = candidate_generate(pose.cuda(), move=move)
    #     pcd_num_ca, gt_depth_list, gt_color_list = headless_render_cadidate(mesh, candidate_generate_list, history=True, v=volu)

        #



        # for i, img in enumerate(gt_color_list):
        #     plt.imshow(np.asarray(img))
        #     plt.axis('off')
        #     plt.savefig(os.path.join(gt_folder_pose, f"rgb_{i}.jpg"),bbox_inches='tight',pad_inches = 0)
        # for i, img in enumerate(gt_depth_list):
        #     plt.imshow(np.asarray(img))
        #     plt.axis('off')
        #     plt.savefig(os.path.join(gt_folder_pose, f"depth_{i}.jpg"),bbox_inches='tight',pad_inches = 0)

    return 0
if __name__ == '__main__':
    gt_nbv_generation(input_folder, use_pcd=True)
