import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from candidate_renderer import *
import open3d as o3d
import cv2

image_dir = "Datasets/scannet/scans"


def metrics_generation(image_dir):
  dirs = sorted(os.listdir(image_dir))
  # for dir in dirs:
  #   if os.path.isfile(os.path.join(image_dir,dir)):
  #     dirs.remove(dir)
  
  for dir in dirs[4:]:
    render_dir = os.path.join(image_dir, dir,"render")
    frames = sorted(os.listdir(render_dir))
    for frame in frames:
      print(f"Now is processing frame {frame} of {dir}...")
      frame_dir = os.path.join(render_dir,frame)
      candidate = ["y+degree","y-degree","x+degree","x-degree"]

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

      for i,ca in enumerate(candidate):

        render_color_ca = cv2.imread(os.path.join(frame_dir,f"rgb_{ca}.jpg"),0)
        render_depth_ca = cv2.imread(os.path.join(frame_dir,f"depth_{ca}.png"),0)

        # fft, size = 60
        # the lower, the blurrier
        render_color_ca_fft, render_color_fft_metric = detect_blur_fft(render_color_ca, size=60)
        render_depth_ca_fft, render_depth_fft_metric = detect_blur_fft(render_depth_ca, size=60)

        # fft, size = 20
        # the lower, the blurrier
        render_color_ca_fft_20, render_color_fft_metric_20 = detect_blur_fft(render_color_ca, size=20)
        render_depth_ca_fft_20, render_depth_fft_metric_20 = detect_blur_fft(render_depth_ca, size=20)

        # fft, size = 10
        # the lower, the blurrier
        render_color_ca_fft_10, render_color_fft_metric_10 = detect_blur_fft(render_color_ca, size=10)
        render_depth_ca_fft_10, render_depth_fft_metric_10 = detect_blur_fft(render_depth_ca, size=10)

        # fft, size = 100
        # the lower, the blurrier
        render_color_ca_fft_100, render_color_fft_metric_100 = detect_blur_fft(render_color_ca, size=100)
        render_depth_ca_fft_100, render_depth_fft_metric_100 = detect_blur_fft(render_depth_ca, size=100)

        # laplacian
        # the lower, the blurrier
        render_color_lap_metric = cv2.Laplacian(np.float64(render_color_ca), cv2.CV_64F).var()
        render_depth_lap_metric = cv2.Laplacian(render_depth_ca, cv2.CV_64F).var()

        # Blurred Region Detection using Singular Value Decomposition (SVD)
        render_color_SVD = get_blur_degree(render_color_ca)
        render_depth_SVD = get_blur_degree(render_depth_ca)

        # brenner
        render_color_brenner = brenner(render_color_ca)
        render_depth_brenner = brenner(render_depth_ca)

        # SMD
        render_color_SMD = SMD(render_color_ca)
        render_depth_SMD = SMD(render_depth_ca)

        # SMD2
        render_color_SMD2 = SMD2(render_color_ca)
        render_depth_SMD2 = SMD2(render_depth_ca)

        # variance
        render_color_variance = variance(render_color_ca)
        render_depth_variance = variance(render_depth_ca)

        # energy
        render_color_energy = energy(render_color_ca)
        render_depth_energy = energy(render_depth_ca)

        # Vollath
        render_color_Vollath = Vollath(render_color_ca)
        render_depth_Vollath = Vollath(render_depth_ca)

        # NRSS
        render_color_NRSS = NRSS(render_color_ca)
        render_depth_NRSS = NRSS(render_depth_ca)

        # tenengrad
        render_color_tenengrad = tenengrad(render_color_ca)
        render_depth_tenengrad = tenengrad(render_depth_ca)

        # GLVN
        render_color_GLVN = normalizedGraylevelVariance(render_color_ca)
        render_depth_GLVN = normalizedGraylevelVariance(render_depth_ca)


        # varianceOfLaplacian
        render_color_LAPV = varianceOfLaplacian(render_color_ca)
        render_depth_LAPV = varianceOfLaplacian(render_depth_ca)

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
      df.to_csv(os.path.join(frame_dir, f'{frame}.csv'))
        


metrics_generation(image_dir)




