import cv2
import numpy as np


# def fill_hole(im_in):
#     # 复制 im_in 图像
#     im_floodfill = im_in.copy()
#
#     # Mask 用于 floodFill，官方要求长宽+2
#     h, w = im_in.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#
#     # floodFill函数中的seedPoint对应像素必须是背景
#     isbreak = False
#     for i in range(im_floodfill.shape[0]):
#         for j in range(im_floodfill.shape[1]):
#             if (im_floodfill[i][j] == 0):
#                 seedPoint = (i, j)
#                 isbreak = True
#                 break
#         if (isbreak):
#             break
#
#     # 得到im_floodfill 255填充非孔洞值
#     cv2.floodFill(im_floodfill, mask, seedPoint, 255)
#
#     # 得到im_floodfill的逆im_floodfill_inv
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
#     # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
#     im_out = im_in | im_floodfill_inv
#
#     # 保存结果
#     return im_out

def select_shape_by_area(region, threshold_min, threshold_max, only_max=False):
    region_clone = region.copy()
    region_num, region_connection = cv2.connectedComponents(region_clone)
    count = np.bincount(region_connection.flatten())
    count[0] = 0
    if only_max:
        threshold_min = threshold_max = max(count)
    flag = [255 if cnt >= threshold_min and cnt <= threshold_max else 0 for cnt in count]
    return np.vectorize(lambda x: flag[x])(region_connection).astype(np.uint8)


def fill_hole_shape_by_area(region, threshold_min, threshold_max):
    region_inv = 255 - region
    holes = select_shape_by_area(region_inv, threshold_min, threshold_max)
    return region + holes


def fill_holes(region):
    return fill_hole_shape_by_area(region, 1, region.size())
