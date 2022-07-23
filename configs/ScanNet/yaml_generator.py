import os
import glob
import numpy as np
import open3d as o3d
import yaml
mesh = o3d.io.read_triangle_mesh("Datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply")

config_folder = "configs/ScanNet"
mesh_folder = "Datasets/scannet/scans"

for root, dirs, files in os.walk(mesh_folder):
  dirs = sorted(dirs)
  for name in dirs:
    mesh_dir = glob.glob(os.path.join(root,name,"*.ply"))[0]
    mesh = o3d.io.read_triangle_mesh(mesh_dir)
    min = np.min(np.asarray(mesh.vertices),0) - 2.0
    max = np.max(np.asarray(mesh.vertices),0) + 2.0
    bound = np.round(np.vstack([min,max]))
    print(name)
    bound_list = [list(bound[:,0]),list(bound[:,1]),list(bound[:,2])]
    print(bound_list)
    # txt = {
    #   'inherit_from': 'configs/ScanNet/scannet.yaml',
    #   'mapping':
    #   {'bound': '[]',
    #     'marching_cubes_bound': '[[0],[1]]'},
    #   'data':
    #   {'input_folder': f'Datasets/scannet/scans/{name}',
    #     'output': f'output/scannet/scans/{name}'}
    # }
    # with open(os.path.join(config_folder,f"{name}.yaml"), 'w', encoding="utf8") as f:
    #   yaml.dump(txt,f)