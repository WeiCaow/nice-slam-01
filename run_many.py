import os
import glob

image_dir = "Datasets/scannet/scans"

dirs = sorted(os.listdir(image_dir))
for dir in dirs:
  if not os.path.isdir(os.path.join(image_dir,dir)):
    dirs.remove(dir)

for dir in sorted(dirs)[108:]:
  print(dir)
  config = os.path.join("configs/ScanNet",f"{dir}.yaml")
  
  os.system(f'python run.py {config}')             

# 50_02 还没有处理: 删掉了
# 测试一下51_00

