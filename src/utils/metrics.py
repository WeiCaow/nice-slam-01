import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
from src.utils.candidate_renderer import *
import open3d as o3d
import cv2
import pandas as pd

class metrics(object):
    """
    Generate values of different metrics
    """
    def __init__(self, metrics_dir, renderer, verbose, device='cuda:0'):
        self.device = device
        self.metrics_dir = metrics_dir
        self.verbose = verbose
        self.renderer = renderer
        os.makedirs(f'{metrics_dir}', exist_ok=True)

    def gen(self, idx,gt_depth, c2w, c, decoders):

        candidate_generate_list = candidate_generate(c2w, move=10)
        candidate = ["x+Degree", "x-Degree", "y-Degree", "y+Degree"]
        render_depth_list = []
        render_color_list = []

        for i, ca in enumerate(candidate):
            depth, uncertainty, color = self.renderer.render_img(
                c,
                decoders,
                candidate_generate_list[i + 1],
                self.device,
                stage='color',
                gt_depth=gt_depth
                )

            uncertainty_np = uncertainty.detach().cpu().numpy()
            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()
            color_np = np.clip(color_np, 0, 1)*255

            
            # cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            save_dir = os.path.join(self.metrics_dir, f'{idx:05d}')
            os.makedirs(f'{save_dir}', exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'render_uncertainty_{ca}.png'), uncertainty_np)
            cv2.imwrite(os.path.join(save_dir, f'render_depth_{ca}.png'), cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
            # when you read it remember add -1 to depth to keep one channel
            color_np_cv2 = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'render_rgb_{ca}.jpg'), color_np_cv2)
            render_depth_list.append(depth_np)
            render_color_list.append(color_np)

        df = pd.DataFrame(columns=[
            "FFT(size:10)_RGB","FFT(size:20)_RGB",
            "FFT(size:60)_RGB",
            "FFT(size:100)_RGB",
            "FFT(size:10)_Depth","FFT(size:20)_Depth",
            "FFT(size:60)_Depth",
            "FFT(size:100)_Depth",
            "Lap_RGB","Lap_Depth",
            "brenner_RGB","brenner_Depth",
            "SMD_RGB","SMD_Depth",
            "SMD2_RGB","SMD2_Depth",
            "variance_RGB", "variance_Depth",
            "energy_RGB","energy_Depth",
            "Vollath_RGB", "Vollath_Depth",
            "SVD_RGB","SVD_Depth",
            "NRSS_RGB","NRSS_Depth",
            "tenengrad_RGB","tenengrad_Depth",
            "GLVN_RGB","GLVN_Depth",
            "LAPV_RGB","LAPV_Depth"
            ])

        for i, index in enumerate(candidate):
            render_color_ca = render_color_list[i]
            render_depth_ca = render_depth_list[i]

            # fft, size = 60
            # the lower, the blurrier
            render_color_ca_fft, render_color_fft_metric = detect_blur_fft(render_color_ca, size=60, depth=False)
            render_depth_ca_fft, render_depth_fft_metric = detect_blur_fft(render_depth_ca, size=60, depth=True)

            # fft, size = 20
            # the lower, the blurrier
            render_color_ca_fft_20, render_color_fft_metric_20 = detect_blur_fft(render_color_ca, size=20, depth=False)
            render_depth_ca_fft_20, render_depth_fft_metric_20 = detect_blur_fft(render_depth_ca, size=20, depth=True)

            # fft, size = 10
            # the lower, the blurrier
            render_color_ca_fft_10, render_color_fft_metric_10 = detect_blur_fft(render_color_ca, size=10,depth=False)
            render_depth_ca_fft_10, render_depth_fft_metric_10 = detect_blur_fft(render_depth_ca, size=10,depth=True)

            # fft, size = 100
            # the lower, the blurrier
            render_color_ca_fft_100, render_color_fft_metric_100 = detect_blur_fft(render_color_ca, size=100, depth=False)
            render_depth_ca_fft_100, render_depth_fft_metric_100 = detect_blur_fft(render_depth_ca, size=100, depth=True)

            # laplacian
            # the lower, the blurrier
            render_color_ca_gray = cv2.cvtColor(render_color_ca, 7)
            render_color_lap_metric = cv2.Laplacian(np.float64(render_color_ca_gray), cv2.CV_64F).var()
            render_depth_lap_metric = cv2.Laplacian(render_depth_ca, cv2.CV_64F).var()

            # Blurred Region Detection using Singular Value Decomposition (SVD)
            render_color_SVD = get_blur_degree(render_color_ca, depth=False)
            render_depth_SVD = get_blur_degree(render_depth_ca, depth=True)

            # brenner
            render_color_brenner = brenner(render_color_ca, depth=False)
            render_depth_brenner = brenner(render_depth_ca, depth=True)

            # SMD
            render_color_SMD = SMD(render_color_ca, depth=False)
            render_depth_SMD = SMD(render_depth_ca, depth=True)

            # SMD2
            render_color_SMD2 = SMD2(render_color_ca, depth=False)
            render_depth_SMD2 = SMD2(render_depth_ca, depth=True)

            # variance
            render_color_variance = variance(render_color_ca, depth=False)
            render_depth_variance = variance(render_depth_ca, depth=True)

            # energy
            render_color_energy = energy(render_color_ca, depth=False)
            render_depth_energy = energy(render_depth_ca, depth=True)

            # Vollath
            render_color_Vollath = Vollath(render_color_ca, depth=False)
            render_depth_Vollath = Vollath(render_depth_ca, depth=True)

            # NRSS
            render_color_NRSS = NRSS(render_color_ca, depth=False)
            render_depth_NRSS = NRSS(render_depth_ca, depth=True)

            # tenengrad
            render_color_tenengrad = tenengrad(render_color_ca, depth=False)
            render_depth_tenengrad = tenengrad(render_depth_ca, depth=True)

            # GLVN
            render_color_GLVN = normalizedGraylevelVariance(render_color_ca, depth=False)
            render_depth_GLVN = normalizedGraylevelVariance(render_depth_ca, depth=True)


            # varianceOfLaplacian
            render_color_LAPV = varianceOfLaplacian(render_color_ca, depth=False)
            render_depth_LAPV = varianceOfLaplacian(render_depth_ca, depth=True)

            s = pd.Series([
                render_color_fft_metric_10, render_color_fft_metric_20,
                render_color_fft_metric,
                render_color_fft_metric_100,
                render_depth_fft_metric_10, render_depth_fft_metric_20,
                render_depth_fft_metric,
                render_depth_fft_metric_100,
                render_color_lap_metric, render_depth_lap_metric,
                render_color_brenner,render_depth_brenner,
                render_color_SMD,render_depth_SMD,
                render_color_SMD2,render_depth_SMD2,
                render_color_variance, render_depth_variance,
                render_color_energy,render_depth_energy,
                render_color_Vollath, render_depth_Vollath,
                render_color_SVD, render_depth_SVD,
                render_color_NRSS,render_depth_NRSS,
                render_color_tenengrad,render_depth_tenengrad,
                render_color_GLVN,render_depth_GLVN,
                render_color_LAPV,render_depth_LAPV])
            s.index = [
                "FFT(size:10)_RGB","FFT(size:20)_RGB",
                "FFT(size:60)_RGB",
                "FFT(size:100)_RGB",
                "FFT(size:10)_Depth","FFT(size:20)_Depth",
                "FFT(size:60)_Depth",
                "FFT(size:100)_Depth",
                "Lap_RGB","Lap_Depth",
                "brenner_RGB","brenner_Depth",
                "SMD_RGB","SMD_Depth",
                "SMD2_RGB","SMD2_Depth",
                "variance_RGB", "variance_Depth",
                "energy_RGB","energy_Depth",
                "Vollath_RGB", "Vollath_Depth",
                "SVD_RGB","SVD_Depth",
                "NRSS_RGB","NRSS_Depth",
                "tenengrad_RGB","tenengrad_Depth",
                "GLVN_RGB","GLVN_Depth",
                "LAPV_RGB","LAPV_Depth"
            ]
            df = df.append(s, ignore_index=True)
        df.to_csv(os.path.join(self.metrics_dir, f'{idx:05d}.csv'))