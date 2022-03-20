import cv2
import numpy as np

def cv_imread(file_path, flag=-1):
    """
    解决cv包含中文路径的问题
    :param file_path:  路径
    :param flag:
    :return:
    """
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
    return cv_img


def concatImage(images, horizontal=True, scale=1, background=128, gap_len=5):
    """

    :param images:
    :param horizontal:
    :param scale:
    :param background:
    :param gap_width:
    :return: np.uint8
    """
    h, w = images[0].shape[:2]
    h, w = int(scale*h), int(scale*w)
    gap = np.zeros((h, gap_len), dtype=np.uint8) if horizontal else np.zeros((gap_len, w), dtype=np.uint8)
    gap += background
    image_list = []
    for image in images:
        image_list.append(cv2.resize(image, dsize=(w, h)))
        image_list.append(gap.copy())

    image_res = np.concatenate(image_list, axis=1 if horizontal else 0)
    return image_res