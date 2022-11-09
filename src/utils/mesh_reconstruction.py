import numpy as np
import open3d as o3d
import math
import os
import copy
import cv2
import torch
import glob
import argparse
import sys
sys.path.append(".") 
from src import config
from src.utils.datasets import get_dataset

parser = argparse.ArgumentParser(
    description='Arguments for running the NICE-SLAM/iMAP*.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--input_folder', type=str,
                    help='input folder, this have higher priority, can overwrite the one in config file')
parser.add_argument('--output', type=str,
                    help='output folder, this have higher priority, can overwrite the one in config file')
nice_parser = parser.add_mutually_exclusive_group(required=False)
nice_parser.add_argument('--nice', dest='nice', action='store_true')
nice_parser.add_argument('--imap', dest='nice', action='store_false')
parser.set_defaults(nice=True)
args = parser.parse_args()

cfg = config.load_config(
    args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
frame_reader = get_dataset(cfg, args, 1)


camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620/2-0.5 ,cy=460/2-0.5)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    

for idx, gt_color, gt_depth, gt_c2w in frame_reader:
  if idx%50 == 0:  
    print(f"{args.config[-17:-5]} {idx}")
    gt_c2w = np.array(gt_c2w.cpu())
    gt_c2w[:3, 1] *= -1
    gt_c2w[:3, 2] *= -1
    gt_c2w = np.linalg.inv(gt_c2w)

    depth = o3d.geometry.Image(np.array(gt_depth.cpu()).astype(np.float32))
    color = o3d.geometry.Image(np.array(
        (np.array(gt_color.cpu()) * 255).astype(np.uint8)))

    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume.integrate(rgbd,camera_intrinsics, gt_c2w)

    pcd = volume.extract_point_cloud()

    if idx > len(frame_reader) - 200:
        o3d.io.write_point_cloud(f"output/mesh_recon/{args.config[-17:-5]}.ply",pcd,print_progress=True)
        break


