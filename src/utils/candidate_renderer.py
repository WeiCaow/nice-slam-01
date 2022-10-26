import numpy as np
import open3d as o3d
import math
import os
import copy
import cv2
import torch
from skimage.metrics import structural_similarity

#brenner梯度函数计算
def brenner(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

#SMD梯度函数计算
def SMD(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

#SMD2梯度函数计算
def SMD2(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

#方差函数计算
def variance(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

#energy函数计算
def energy(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

#Vollath函数计算
def Vollath(img, depth=True):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    if not depth:
        img = cv2.cvtColor(img, 7)
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def get_blur_degree(img, sv_num=10, depth = True):
    """
    input: image_file, sv_num=10 output: blur degree [0(clear)-1(blur)]
    https://github.com/fled/blur_detection

    """
    if not depth:
        img = cv2.cvtColor(img, 7)
    # img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv

def detect_blur_fft(image, size=60, depth = True):

    if not depth:
        image = cv2.cvtColor(image, 7)
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # output for visualization
    magnitude = 20 * np.log(np.abs(fftShift))

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude_0 = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude_0)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value

    return magnitude, mean


# NRSS
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def getBlock(G,Gr):
    (h, w) = G.shape
    G_blk_list = []
    Gr_blk_list = []
    sp = 6
    for i in range(sp):
        for j in range(sp):
            G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            G_blk_list.append(G_blk)
            Gr_blk_list.append(Gr_blk)
    sum = 0
    for i in range(sp*sp):
        mssim = structural_similarity(G_blk_list[i], Gr_blk_list[i])
        sum = mssim + sum
    nrss = 1-sum/(sp*sp*1.0)
    return nrss

def NRSS(image, depth=True):
    if not depth:
        image = cv2.cvtColor(image, 7)
    #高斯滤波
    Ir = cv2.GaussianBlur(image,(7,7),0)
    G = sobel(image)
    Gr = sobel(Ir)
    blocksize = 8
    ## 获取块信息
    return getBlock(G, Gr)

def tenengrad(img, depth=True, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    if not depth:
        img = cv2.cvtColor(img, 7)
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv2.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    return mn


def normalizedGraylevelVariance(img, depth=True):
    
    ''''GLVN' algorithm (Santos97)'''
    if not depth:
        img = cv2.cvtColor(img, 7)
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0]**2 / mean[0]
    return s[0]
    
def varianceOfLaplacian(img, depth=True):
    ''''LAPV' algorithm (Pech2000)'''
    if not depth:
        img = cv2.cvtColor(img, 7)
    lap = cv2.Laplacian(img, ddepth=-1)#cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0]**2
    return s[0]

def pcd_integration(pcd,pcd_sum):
    # print(np.array(pcd_sum.points).shape)
    # print(np.array(pcd.points).shape)
    dists = pcd.compute_point_cloud_distance(pcd_sum)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.15e-1)[0]
    pcd_sum_new = pcd.select_by_index(ind)+pcd_sum
    # print(np.array(pcd_sum.points).shape)
    return pcd_sum_new

def headless_render(mesh, camera_extrinsics,width = 620, height = 480):
    '''
        input: mesh of the scene, image width, image height, camera intrinsics, camera extrinsics: list
        合成一个list中所有pose的volume
        返回每一步的点云的点数
    '''

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620/2-0.5 ,cy=460/2-0.5)
    
    
    pcd_num = []

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    for i in range(len(camera_extrinsics)):

        camera_extrinsics_copy = np.copy(camera_extrinsics[i])
        camera_extrinsics_copy[:3, 1] *= -1
        camera_extrinsics_copy[:3, 2] *= -1
        camera_extrinsics_copy = np.linalg.inv(camera_extrinsics_copy)

        render = o3d.visualization.rendering.OffscreenRenderer(620, 460, headless=True)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1, 1, 1, 1]
        mtl.shader = "defaultUnlit"
        render.scene.set_background([0, 0, 0, 0])
        render.scene.add_geometry("model", mesh, mtl)
        render.setup_camera(camera_intrinsics, camera_extrinsics_copy)
        dimg = render.render_to_depth_image(True)
        cimg = render.render_to_image()

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cimg, dimg, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

        volume.integrate(rgbd,camera_intrinsics, camera_extrinsics_copy)
        pcd = volume.extract_point_cloud()
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        num_pcd= np.asarray(pcd.points).shape[0]
        pcd_num.append(num_pcd)
 
    return pcd_num, volume

def headless_add_one_render(mesh, pose=None, v=None):

    volume = copy.deepcopy(v)
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620 / 2 - 0.5,
                                     cy=460 / 2 - 0.5)

    camera_extrinsics_copy = np.copy(pose)
    camera_extrinsics_copy[:3, 1] *= -1
    camera_extrinsics_copy[:3, 2] *= -1
    camera_extrinsics_copy = np.linalg.inv(camera_extrinsics_copy)

    render = o3d.visualization.rendering.OffscreenRenderer(620, 460, headless=True)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1, 1, 1, 1]
    mtl.shader = "defaultUnlit"
    render.scene.set_background([0, 0, 0, 0])
    render.scene.add_geometry("model", mesh, mtl)
    render.setup_camera(camera_intrinsics, camera_extrinsics_copy)
    dimg = render.render_to_depth_image(True)
    cimg = render.render_to_image()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        cimg, dimg, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume.integrate(rgbd, camera_intrinsics, camera_extrinsics_copy)
    pcd = volume.extract_point_cloud()
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    num_pcd = np.asarray(pcd.points).shape[0]
    return num_pcd, volume

def headless_render_cadidate(mesh, camera_extrinsics_tensor, width = 620, height = 480, history = False, v = None, path=None):
    '''
        input: mesh of the scene, image width, image height, camera intrinsics, camera extrinsics: list
        history: 是否考虑以前已经见到过的点云
        
        返回每一个candidate渲染出的点云和之前的点云之和的点云数量，两个candidate就会有两个点云之和

    '''

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620/2-0.5, cy=460/2-0.5)

    candidate = ["x+0.2", "x-0.2", "y+0.2", "y-0.2", "z+0.2", "z-0.2"]
    
    pcd_num = []

    # os.makedirs(path, exist_ok=True)
    camera_extrinsics = copy.deepcopy(camera_extrinsics_tensor)
    for i in range(len(camera_extrinsics)):
        camera_extrinsics[i] = camera_extrinsics[i].detach().cpu().numpy()
        camera_extrinsics[i][:3, 1] *= -1
        camera_extrinsics[i][:3, 2] *= -1
        camera_extrinsics[i] = np.linalg.inv(camera_extrinsics[i])
    depth_ca_list = []
    color_ca_list = []

    for i in range(1, len(camera_extrinsics)):
        render = o3d.visualization.rendering.OffscreenRenderer(
            620, 460, headless=True)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1, 1, 1, 1]
        mtl.shader = "defaultUnlit"
        render.scene.set_background([0, 0, 0, 0])
        render.scene.add_geometry("model", mesh, mtl)
        if not history:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 / 512.0,
                sdf_trunc=0.04,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

            render.setup_camera(camera_intrinsics, camera_extrinsics[0])
            dimg = render.render_to_depth_image(True)
            cimg = render.render_to_image()
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(cimg, dimg, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)
            volume.integrate(rgbd,camera_intrinsics,camera_extrinsics[0])
        else:
            volume = copy.deepcopy(v)
        
        render.setup_camera(camera_intrinsics,camera_extrinsics[i])
        dimg_ca = render.render_to_depth_image(True)
        cimg_ca = render.render_to_image()
        depth_ca_list.append(dimg_ca)
        color_ca_list.append(cimg_ca)
        # o3d.io.write_image(f'{path}/gt_depth_{candidate[i-1]}.png', dimg_ca)
        # o3d.io.write_image(f'{path}/gt_color_{candidate[i-1]}.jpg', cimg_ca)

        # import matplotlib.pyplot as plt
        # a = np.asarray(render.render_to_depth_image(True))
        # plt.imshow(a)
        # plt.show()

        rgbd_ca = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cimg_ca, dimg_ca, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

        volume.integrate(rgbd_ca,camera_intrinsics,camera_extrinsics[i])
        pcd = volume.extract_point_cloud()
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        num_pcd = np.asarray(pcd.points).shape[0]
        pcd_num.append(num_pcd)
 
    return pcd_num, depth_ca_list, color_ca_list

def candidate_generate(camera_extrinsics_origin, move = int):
    '''  ["origin","x+0.2","x-0.2","y+0.2","y-0.2","z+0.2","z-0.2"]
    '''

    camera_extrinsics_list = []
    origin = copy.deepcopy(camera_extrinsics_origin)
    # origin[:, 1] *= -1
    # origin[:, 2] *= -1
    # camera_extrinsics_list.append(origin)

    for i in range(4):
        camera_extrinsics_ca = copy.deepcopy(origin)
        if i==0:
            # camera_extrinsics_ca[0, -1] = camera_extrinsics_ca[0, -1] + move
            camera_extrinsics_ca[:3, :3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([math.radians(move), 0, 0])).float().cuda()

        elif i==1:
            # camera_extrinsics_ca[0, -1] = camera_extrinsics_ca[0, -1] - move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([-math.radians(move),0,0])).float().cuda()

        elif i==2:
            # camera_extrinsics_ca[1, -1] = camera_extrinsics_ca[1, -1] + move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.radians(move),0])).float().cuda()

        elif i==3:
            # camera_extrinsics_ca[1, -1] = camera_extrinsics_ca[1, -1] - move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([0, -math.radians(move),0])).float().cuda()

        # elif i==4:
        #     # camera_extrinsics_ca[2, -1] = camera_extrinsics_ca[2, -1] + move
        #     camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([0,0, math.radians(move)])).float().cuda()
        #
        # elif i==5:
        #     # camera_extrinsics_ca[2, -1] = camera_extrinsics_ca[2, -1] - move
        #     camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle([0,0, -math.radians(move)])).float().cuda()

        
        # if not c2w:
        #     camera_extrinsics_list.append(np.linalg.inv(camera_extrinsics_ca))
        # else:
        camera_extrinsics_list.append(camera_extrinsics_ca)

    # pcd_num = headless_render_cadidate(mesh,camera_extrinsics_list, history=False, volume=None)

    return camera_extrinsics_list

def candidate_generate_np(camera_extrinsics_origin, move = int):
    '''  ["origin","x+0.2","x-0.2","y+0.2","y-0.2","z+0.2","z-0.2"]
    '''

    camera_extrinsics_list = []
    origin = copy.deepcopy(camera_extrinsics_origin)
    # origin[:, 1] *= -1
    # origin[:, 2] *= -1

    for i in range(4):
        camera_extrinsics_ca = copy.deepcopy(origin)
        if i==0:
            # camera_extrinsics_ca[0, -1] = camera_extrinsics_ca[0, -1] + move
            camera_extrinsics_ca[:3, :3] = camera_extrinsics_ca[:3,:3] @ o3d.geometry.get_rotation_matrix_from_axis_angle([math.radians(move), 0, 0])
        elif i==1:
            # camera_extrinsics_ca[0, -1] = camera_extrinsics_ca[0, -1] - move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ o3d.geometry.get_rotation_matrix_from_axis_angle([-math.radians(move),0,0])

        elif i==2:
            # camera_extrinsics_ca[1, -1] = camera_extrinsics_ca[1, -1] + move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.radians(move),0])

        elif i==3:
            # camera_extrinsics_ca[1, -1] = camera_extrinsics_ca[1, -1] - move
            camera_extrinsics_ca[:3,:3] = camera_extrinsics_ca[:3,:3] @ o3d.geometry.get_rotation_matrix_from_axis_angle([0, -math.radians(move),0])


        camera_extrinsics_list.append(camera_extrinsics_ca)

    return camera_extrinsics_list

def headless_pcd_render(mesh, camera_extrinsics):
    '''
        input: mesh of the scene, image width, image height
        return:
        a list of the number of points in the integrated candidate point clouds;
        a final point cloud
    '''

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620 / 2 - 0.5,
                                     cy=460 / 2 - 0.5)

    pcd_num = []

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_extrinsics)):
        print(f"{i} frame of total {len(camera_extrinsics)} frames")
        camera_extrinsics_copy = np.copy(camera_extrinsics[i])
        # 如果直接从dataset里提取pose需要-1，c2w不需要
        # camera_extrinsics_copy[:3, 1] *= -1
        # camera_extrinsics_copy[:3, 2] *= -1
        camera_extrinsics_copy = np.linalg.inv(camera_extrinsics_copy)

        render = o3d.visualization.rendering.OffscreenRenderer(
            620, 460, headless=True)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1, 1, 1, 1]
        mtl.shader = "defaultUnlit"
        render.scene.set_background([0, 0, 0, 0])
        render.scene.add_geometry("model", mesh, mtl)
        render.setup_camera(camera_intrinsics, camera_extrinsics_copy)


        dimg = render.render_to_depth_image( True )
        # o3d.io.write_image("o3d/depth_{}.png".format(i), dimg)
        cimg = render.render_to_image()
        # o3d.io.write_image("o3d/rgb_{}.jpg".format(i), cimg)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cimg, dimg, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

        volume.integrate(rgbd, camera_intrinsics, camera_extrinsics_copy)
        pcd = volume.extract_point_cloud()
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        num_pcd = np.asarray(pcd.points).shape[0]
        pcd_num.append(num_pcd)

    return pcd_num, pcd

def headless_pcd_ca_render(mesh, camera_extrinsics, pcd=None):

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620/2-0.5 ,cy=460/2-0.5)

    pcd_ca_num = []
    for i in range(len(camera_extrinsics)):
        camera_extrinsics[i] = camera_extrinsics[i]
        camera_extrinsics[i][:3, 1] *= -1
        camera_extrinsics[i][:3, 2] *= -1
        camera_extrinsics[i] = np.linalg.inv(camera_extrinsics[i])

    for i in range(len(camera_extrinsics)):
        render = o3d.visualization.rendering.OffscreenRenderer(
            620, 460, headless=True)
        mtl = o3d.visualization.rendering.MaterialRecord()
        # mtl.base_color = [1, 1, 1, 1]
        # mtl.base_color = [0, 0, 0, 0]
        # mtl.shader = "depth"
        mtl.shader = "defaultUnlit"
        # render.scene.set_background([0, 0, 0, 0])
        render.scene.add_geometry("model", mesh, mtl)


        pcd_ca = copy.deepcopy(pcd)

        render.setup_camera(camera_intrinsics,camera_extrinsics[i])
        dimg_ca = render.render_to_depth_image(True)
        cimg_ca = render.render_to_image()

        rgbd_ca = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cimg_ca, dimg_ca, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

        cimg_ca = np.asarray(cimg_ca)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        volume.integrate(rgbd_ca,camera_intrinsics,camera_extrinsics[i])

        pcd_ca_new = volume.extract_point_cloud()
        pcd_ca_new_in = pcd_integration(pcd_ca_new,pcd_ca)

        # pcd_ca_new = pcd_ca_new + pcd_ca

        # pcd_ca_new = pcd_ca_new.voxel_down_sample(voxel_size=0.01)
        num_pcd = np.asarray(pcd_ca_new_in.points).shape[0]
        pcd_ca_num.append(num_pcd)

    return pcd_ca_num

def headless_add_one_pcd_render(mesh, pose=None, p=None):

    pcd_origin = copy.deepcopy(p)
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(width=620, height=460, fx=577.590698, fy=578.729797, cx=620 / 2 - 0.5,
                                     cy=460 / 2 - 0.5)

    camera_extrinsics_copy = np.copy(pose)
    camera_extrinsics_copy[:3, 1] *= -1
    camera_extrinsics_copy[:3, 2] *= -1
    camera_extrinsics_copy = np.linalg.inv(camera_extrinsics_copy)

    render = o3d.visualization.rendering.OffscreenRenderer(620, 460, headless=True)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1, 1, 1, 1]
    mtl.shader = "defaultUnlit"
    render.scene.set_background([0, 0, 0, 0])
    render.scene.add_geometry("model", mesh, mtl)
    render.setup_camera(camera_intrinsics, camera_extrinsics_copy)
    dimg = render.render_to_depth_image(True)
    cimg = render.render_to_image()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        cimg, dimg, depth_scale=1, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    volume.integrate(rgbd, camera_intrinsics, camera_extrinsics_copy)
    pcd = volume.extract_point_cloud()
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    new_pcd = pcd_integration(pcd,pcd_origin)
    # new_pcd = pcd_origin + pcd
    # new_pcd = new_pcd.voxel_down_sample(voxel_size=0.05)
    num_pcd = np.asarray(new_pcd.points).shape[0]
    return num_pcd, new_pcd


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh(
        "/home/cao/Desktop/nice-slam/Datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply")
    test = [torch.Tensor([[-9.5542e-01, -1.1962e-01, 2.6993e-01, 2.6558e+00],
                      [2.9525e-01, -3.8834e-01, 8.7294e-01, 2.9816e+00],
                      [4.0800e-04, 9.1372e-01, 4.0634e-01, 1.3686e+00],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]),
            torch.Tensor([[-0.9516, -0.1204, 0.2828, 2.6555],
                      [0.3073, -0.3925, 0.8669, 2.9814],
                      [0.0067, 0.9118, 0.4105, 1.3619],
                      [0.0000, 0.0000, 0.0000, 1.0000]]),
            torch.Tensor([[-0.9474, -0.1260, 0.2942, 2.6494],
                      [0.3199, -0.4012, 0.8583, 2.9786],
                      [0.0099, 0.9073, 0.4204, 1.3654],
                      [0.0000, 0.0000, 0.0000, 1.0000]])]
    c2w = torch.Tensor([[-9.4550e-01, -1.3620e-01, 2.9578e-01, 2.6443e+00],
            [3.2563e-01, -4.0164e-01, 8.5595e-01, 2.9783e+00],
            [2.2132e-03, 9.0561e-01, 4.2410e-01, 1.3619e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])
    # n, volu = headless_render(mesh, test)
    # # path = 'output/scannet/scans/scene0000_00/tracking_vis/00005_0000'
    # print("original:",np.asarray(volu.extract_point_cloud().points).shape[0])
    # list = candidate_generate(c2w.cuda(), move=30)
    # # print(list[0])
    # # # print(list[1])
    # voly_t = copy.deepcopy(volu)
    # pcd_num,_,_= headless_render_cadidate(mesh, list, v=voly_t)
    # print(pcd_num)
    # print("after:", np.asarray(volu.extract_point_cloud().points).shape[0])
   