import os
import cv2
import numpy as np
from tqdm import tqdm

def cul_mean_std(img_dirs,):
    r, g, b = [], [], []
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    for img_dir in img_dirs:
        for img_name in tqdm(os.listdir(img_dir)):
            img = cv2.imread(os.path.join(img_dir, img_name), -1)
            # trans_rgb
            img = img[:, :, ::-1]
            r += list(img[:, :, 0].flatten())
            g += list(img[:, :, 1].flatten())
            b += list(img[:, :, 2].flatten())

    r_mean = np.nanmean(r)
    g_mean = np.nanmean(g)
    b_mean = np.nanmean(b)
    r_std = np.nanstd(r)
    g_std = np.nanstd(g)
    b_std = np.nanstd(b)
    print("Mean: r{}, g{}, b{}".format(r_mean, g_mean, b_mean))
    print("Std: r{}, g{}, b{}".format(r_std, g_std, b_std))


if __name__ == "__main__":
    cul_mean_std(img_dirs="/home/gdut502/下载/初赛数据集/train/a_images")

