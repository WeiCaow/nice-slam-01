import os
from shutil import move
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
from src.utils.candidate_renderer import *
import open3d as o3d
import cv2
import pandas as pd
from PIL import Image

class Metrics(object):
    """
    Generate values of different metrics
    """
    def __init__(self, metrics_dir, move, renderer, verbose, device='cuda:0'):
        self.device = device
        self.metrics_dir = metrics_dir
        self.move = move
        self.verbose = verbose
        self.renderer = renderer
        os.makedirs(f'{metrics_dir}', exist_ok=True)

    def gen(self, idx, gt_depth, c2w, c, decoders):

        print("begin rendering with larger FoV...")
        depth_pad, uncertainty_pad, color_pad = self.renderer.render_img_metric(
            c,
            decoders,
            c2w,
            self.device,
            stage='color'
            )

        uncertainty_np_pad = uncertainty_pad.detach().cpu().numpy()
        depth_np_pad = depth_pad.detach().cpu().numpy()
        color_np_pad = color_pad.detach().cpu().numpy()
        color_np_pad = np.clip(color_np_pad, 0, 1)

        depth, uncertainty, color = self.renderer.render_img(
            c,
            decoders,
            c2w,
            self.device,
            stage='color',
            gt_depth = gt_depth
            )

        uncertainty_np = uncertainty.detach().cpu().numpy()
        depth_np = depth.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        color_np = np.clip(color_np, 0, 1)

        h = int((color_np_pad.shape[0] - color_np.shape[0])/2)
        w = int((color_np_pad.shape[1] - color_np.shape[1])/2)

        color_np_pad[h : h+color_np.shape[0], w : w+color_np.shape[1]] = color_np
        depth_np_pad[h : h+color_np.shape[0], w : w+color_np.shape[1]] = depth_np
        uncertainty_np_pad[h : h+color_np.shape[0], w : w+color_np.shape[1]] = uncertainty_np

        save_dir = os.path.join(self.metrics_dir, f'{idx:05d}')
        os.makedirs(f'{save_dir}', exist_ok=True)
        
        np.save(os.path.join(save_dir, f'render_depth_large_scale.npy'),depth_np_pad)
        np.save(os.path.join(save_dir, f'render_rgb_large_scale.npy'),color_np_pad)
        np.save(os.path.join(save_dir, f'render_uncertainty_large_scale.npy'),uncertainty_np_pad)
        
        plt.imsave(os.path.join(save_dir, f'render_rgb_bigFoV.png'),color_np_pad)
        plt.imsave(os.path.join(save_dir, f'render_depth_bigFoV.png'),depth_np_pad)
        plt.imsave(os.path.join(save_dir, f'render_uncertainty_bigFoV.png'),uncertainty_np_pad)


        # print(f"begin rendering with candidate with each {self.move} degree rotation...")
        # candidate_generate_list = candidate_generate(c2w, move=self.move)
        # candidate = ["x+Degree", "x-Degree", "y-Degree", "y+Degree"]
        # render_depth_list = []
        # render_color_list = []

        # for i, ca in enumerate(candidate):   
        #     depth, uncertainty, color = self.renderer.render_img_ca(
        #     c,
        #     decoders,
        #     c2w,
        #     candidate_generate_list[i + 1],
        #     self.device,
        #     stage='color',
        #     gt_depth=gt_depth
        #     )

        #     depth_np = depth.detach().cpu().numpy()
        #     color_np = color.detach().cpu().numpy()
        #     color_np = np.clip(color_np, 0, 1)

        #     render_color_list.append(color_np)
        #     render_depth_list.append(depth_np)

        #     plt.imsave(os.path.join(save_dir, f'render_rgb_{ca}.png'),color_np)
        #     plt.imsave(os.path.join(save_dir, f'render_depth_{ca}.png'),depth_np)

        # df = pd.DataFrame(columns=[
        #     "FFT(size:10)_RGB","FFT(size:20)_RGB",
        #     "FFT(size:60)_RGB",
        #     "FFT(size:100)_RGB",
        #     "FFT(size:10)_Depth","FFT(size:20)_Depth",
        #     "FFT(size:60)_Depth",
        #     "FFT(size:100)_Depth",
        #     "Lap_RGB","Lap_Depth",
        #     "brenner_RGB","brenner_Depth",
        #     "SMD_RGB","SMD_Depth",
        #     "SMD2_RGB","SMD2_Depth",
        #     "variance_RGB", "variance_Depth",
        #     "energy_RGB","energy_Depth",
        #     "Vollath_RGB", "Vollath_Depth",
        #     "SVD_RGB","SVD_Depth",
        #     "NRSS_RGB","NRSS_Depth",
        #     "tenengrad_RGB","tenengrad_Depth",
        #     "GLVN_RGB","GLVN_Depth",
        #     "LAPV_RGB","LAPV_Depth"
        #     ])

        # for i, index in enumerate(candidate):
        #     render_color_ca = render_color_list[i]
        #     render_depth_ca = render_depth_list[i]

        #     # fft, size = 60
        #     # the lower, the blurrier
        #     render_color_ca_fft, render_color_fft_metric = detect_blur_fft(render_color_ca, size=60, depth=False)
        #     render_depth_ca_fft, render_depth_fft_metric = detect_blur_fft(render_depth_ca, size=60, depth=True)

        #     # fft, size = 20
        #     # the lower, the blurrier
        #     render_color_ca_fft_20, render_color_fft_metric_20 = detect_blur_fft(render_color_ca, size=20, depth=False)
        #     render_depth_ca_fft_20, render_depth_fft_metric_20 = detect_blur_fft(render_depth_ca, size=20, depth=True)

        #     # fft, size = 10
        #     # the lower, the blurrier
        #     render_color_ca_fft_10, render_color_fft_metric_10 = detect_blur_fft(render_color_ca, size=10,depth=False)
        #     render_depth_ca_fft_10, render_depth_fft_metric_10 = detect_blur_fft(render_depth_ca, size=10,depth=True)

        #     # fft, size = 100
        #     # the lower, the blurrier
        #     render_color_ca_fft_100, render_color_fft_metric_100 = detect_blur_fft(render_color_ca, size=100, depth=False)
        #     render_depth_ca_fft_100, render_depth_fft_metric_100 = detect_blur_fft(render_depth_ca, size=100, depth=True)

        #     # laplacian
        #     # the lower, the blurrier
        #     render_color_ca_gray = cv2.cvtColor(render_color_ca, 7)
        #     render_color_lap_metric = cv2.Laplacian(np.float64(render_color_ca_gray), cv2.CV_64F).var()
        #     render_depth_lap_metric = cv2.Laplacian(render_depth_ca, cv2.CV_64F).var()

        #     # Blurred Region Detection using Singular Value Decomposition (SVD)
        #     render_color_SVD = get_blur_degree(render_color_ca, depth=False)
        #     render_depth_SVD = get_blur_degree(render_depth_ca, depth=True)

        #     # brenner
        #     render_color_brenner = brenner(render_color_ca, depth=False)
        #     render_depth_brenner = brenner(render_depth_ca, depth=True)

        #     # SMD
        #     render_color_SMD = SMD(render_color_ca, depth=False)
        #     render_depth_SMD = SMD(render_depth_ca, depth=True)

        #     # SMD2
        #     render_color_SMD2 = SMD2(render_color_ca, depth=False)
        #     render_depth_SMD2 = SMD2(render_depth_ca, depth=True)

        #     # variance
        #     render_color_variance = variance(render_color_ca, depth=False)
        #     render_depth_variance = variance(render_depth_ca, depth=True)

        #     # energy
        #     render_color_energy = energy(render_color_ca, depth=False)
        #     render_depth_energy = energy(render_depth_ca, depth=True)

        #     # Vollath
        #     render_color_Vollath = Vollath(render_color_ca, depth=False)
        #     render_depth_Vollath = Vollath(render_depth_ca, depth=True)

        #     # NRSS
        #     render_color_NRSS = NRSS(render_color_ca, depth=False)
        #     render_depth_NRSS = NRSS(render_depth_ca, depth=True)

        #     # tenengrad
        #     render_color_tenengrad = tenengrad(render_color_ca, depth=False)
        #     render_depth_tenengrad = tenengrad(render_depth_ca, depth=True)

        #     # GLVN
        #     render_color_GLVN = normalizedGraylevelVariance(render_color_ca, depth=False)
        #     render_depth_GLVN = normalizedGraylevelVariance(render_depth_ca, depth=True)


        #     # varianceOfLaplacian
        #     render_color_LAPV = varianceOfLaplacian(render_color_ca, depth=False)
        #     render_depth_LAPV = varianceOfLaplacian(render_depth_ca, depth=True)

        #     s = pd.Series([
        #         render_color_fft_metric_10, render_color_fft_metric_20,
        #         render_color_fft_metric,
        #         render_color_fft_metric_100,
        #         render_depth_fft_metric_10, render_depth_fft_metric_20,
        #         render_depth_fft_metric,
        #         render_depth_fft_metric_100,
        #         render_color_lap_metric, render_depth_lap_metric,
        #         render_color_brenner,render_depth_brenner,
        #         render_color_SMD,render_depth_SMD,
        #         render_color_SMD2,render_depth_SMD2,
        #         render_color_variance, render_depth_variance,
        #         render_color_energy,render_depth_energy,
        #         render_color_Vollath, render_depth_Vollath,
        #         render_color_SVD, render_depth_SVD,
        #         render_color_NRSS,render_depth_NRSS,
        #         render_color_tenengrad,render_depth_tenengrad,
        #         render_color_GLVN,render_depth_GLVN,
        #         render_color_LAPV,render_depth_LAPV])
        #     s.index = [
        #         "FFT(size:10)_RGB","FFT(size:20)_RGB",
        #         "FFT(size:60)_RGB",
        #         "FFT(size:100)_RGB",
        #         "FFT(size:10)_Depth","FFT(size:20)_Depth",
        #         "FFT(size:60)_Depth",
        #         "FFT(size:100)_Depth",
        #         "Lap_RGB","Lap_Depth",
        #         "brenner_RGB","brenner_Depth",
        #         "SMD_RGB","SMD_Depth",
        #         "SMD2_RGB","SMD2_Depth",
        #         "variance_RGB", "variance_Depth",
        #         "energy_RGB","energy_Depth",
        #         "Vollath_RGB", "Vollath_Depth",
        #         "SVD_RGB","SVD_Depth",
        #         "NRSS_RGB","NRSS_Depth",
        #         "tenengrad_RGB","tenengrad_Depth",
        #         "GLVN_RGB","GLVN_Depth",
        #         "LAPV_RGB","LAPV_Depth"
        #     ]
        #     df = df.append(s, ignore_index=True)
        # df.to_csv(os.path.join(save_dir, f'{idx:05d}.csv'))