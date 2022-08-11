import os
import glob
import numpy as np
import open3d as o3d
import yaml
import copy
# from ruamel.yaml import YAML
# mesh = o3d.io.read_triangle_mesh(
#     "Datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply")

config_folder = "configs/ScanNet"
mesh_folder = "Datasets/scannet/scans"

# with open("configs/ScanNet/scene0024_02 copy.yaml", 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
#     print(data_loaded)

for root, dirs, files in os.walk(mesh_folder):
    dirs = sorted(dirs)
    for name in dirs[118:]:
        mesh_dir = glob.glob(os.path.join(root, name, "*_vh_clean.ply"))[0]
        mesh = o3d.io.read_triangle_mesh(mesh_dir)
        min = np.min(np.asarray(mesh.vertices), 0) - 2.0
        max = np.max(np.asarray(mesh.vertices), 0) + 2.0
        bound = np.round(np.vstack([min, max]))
        print(name)
        bound_list = [list(bound[:, 0]), list(bound[:, 1]), list(bound[:, 2])]
        # print(bound_list)
        # data_loaded['mapping']['bound'] = bound_list
        txt = {
          'inherit_from': 'configs/ScanNet/scannet.yaml',
          'mapping':
          {'bound': bound_list,
            'marching_cubes_bound': bound_list},
          'data':
          {'input_folder': f'Datasets/scannet/scans/{name}',
            'output': f'output/scannet/scans/{name}'}
        }

        # yaml = YAML()
        # yaml.default_flow_style=True
      
        with open(os.path.join(config_folder, f"{name}.yaml"), 'w') as f:
            # yaml.dump(txt, f)
            print(txt,file=f)

