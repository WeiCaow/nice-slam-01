from genericpath import isdir
from pickle import FALSE
import numpy as np
from random import sample
import random
import os

generate_data = False
split = False
oversample = True


if generate_data == True:
  image_dir = "Datasets/scannet/scans"
  dirs = sorted(os.listdir(image_dir))
  for dir in dirs:
      if not os.path.isdir(os.path.join(image_dir,dir)):
          dirs.remove(dir)

  f = open("Datasets/scannet/scans/nbv_data.txt", "w")
  for dir in dirs:
      render_dir = os.path.join(image_dir, dir,"render")
      if os.path.isdir(render_dir) == True:
        frames = sorted(os.listdir(render_dir))
        for frame in frames:
            big_FOV = os.path.join(render_dir,frame)
            nbv_dir = os.path.join(image_dir,dir,"nbv",f"{frame}.txt")
            nbv = np.loadtxt(nbv_dir)
            if np.max(nbv) == np.min(nbv):
                break
            else:
                nbv = np.argmax(nbv)
                f.write(str(big_FOV)+" "+str(nbv)+'\n')
  f.close()

if split == True:
  f = open("/home/cao/Desktop/nice-slam-01/Datasets/scannet/scans/nbv_data.txt", "r")
  imgs = []
  labels = []
  for line in f:
      line = line.rstrip()
      words = line.split()
      imgs.append((words[0], int(words[1])))
      labels.append(int(words[1]))
  files_list = imgs
  ratio_train = 0.8 #训练集比例
  ratio_trainval = 0.1 #验证集比例
  ratio_val = 0.1 #测试集比例
  assert (ratio_train + ratio_trainval + ratio_val) == 1.0,'Total ratio Not equal to 1' ##检查总比例是否等于1
  np.random.shuffle(files_list) ##打乱文件列表
  cnt_val = round(len(files_list) * ratio_val ,0)
  cnt_trainval = round(len(files_list) * ratio_trainval ,0)
  cnt_train = len(files_list) - cnt_val - cnt_trainval
  print("val Sample:" + str(cnt_val))
  print("trainval Sample:" + str(cnt_trainval))
  print("train Sample:" + str(cnt_train))

  np.random.shuffle(files_list) ##打乱文件列表
  train_list = []
  trainval_list = []
  val_list = []

  for i in range(int(cnt_train)):
      train_list.append(files_list[i])

  for i in range(int(cnt_train) ,int(cnt_train + cnt_trainval)):
      trainval_list.append(files_list[i])

  for i in range(int(cnt_train + cnt_trainval) ,int(cnt_train + cnt_trainval + cnt_val)):
      val_list.append(files_list[i])


  file = open('Datasets/scannet/scans/nbv_data_train.txt','w')
  for i in range(len(train_list)):   
      # name = str(train_list[i])
      # index = name.rfind('.')
      # name = name[:index]
      file.write(train_list[i][0]+" "+str(train_list[i][1]) + '\n')
  file.close()

  file = open('Datasets/scannet/scans/nbv_data_val.txt','w')
  for i in range(len(trainval_list)):   
      file.write(trainval_list[i][0]+" "+str(trainval_list[i][1]) + '\n')
  file.close()

  file = open('Datasets/scannet/scans/nbv_data_test.txt','w')
  for i in range(len(val_list)):   
      # name = str(val_list[i])
      # index = name.rfind('.')
      # name = name[:index]
      file.write(val_list[i][0]+" "+str(val_list[i][1]) + '\n')
  file.close()


# 试一下手动sample， 然后人工增多

if oversample == True:
  f = open("/home/cao/Desktop/nice-slam-01/Datasets/scannet/scans/nbv_data.txt", "r")
  imgs = []
  labels = []
  up = []
  down = []
  left = []
  right = []

  for line in f:
      line = line.rstrip()
      words = line.split()
      imgs.append((words[0], int(words[1])))
      labels.append(int(words[1]))
      if int(words[1]) == 0:
        up.append((words[0], int(words[1])))
      elif int(words[1]) == 1:
        down.append((words[0], int(words[1])))
      elif int(words[1]) == 2:
        left.append((words[0], int(words[1])))
      elif int(words[1]) == 3:
        right.append((words[0], int(words[1])))
        
  cnts = np.bincount(labels)
  down = random.choices(down,k = cnts[0])
  left = random.choices(left,k = cnts[0])
  right = random.choices(right,k = cnts[0])

  imgs = up+down+left+right

  with open('Datasets/scannet/scans/nbv_data_train.txt','w') as ft:
    for i in range(len(imgs)): 
      ft.write(imgs[i][0]+" "+str(imgs[i][1]) + '\n')

  f = open("/home/cao/Desktop/nice-slam-01/Datasets/scannet/scans/nbv_data_train.txt", "r")
  imgs = []
  labels = []
  for line in f:
      line = line.rstrip()
      words = line.split()
      imgs.append((words[0], int(words[1])))
      labels.append(int(words[1]))
  cnts = np.bincount(labels)



