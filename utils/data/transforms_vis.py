from albumentations import *
import json
import os
import random
from PIL import Image
import cv2

if __name__ == "__main__":
    data_root = "./初赛数据集/train"
    transforms = Compose([
        HorizontalFlip(p=1, always_apply=True)
    ], p=1)
    trainA = json.load(open(os.path.join(data_root, "a_annotations.json")))
    random.shuffle(trainA["images"])
    for image in trainA["images"]:
        img = cv2.imread(os.path.join(data_root, "a_images", image["file_name"]))
        img_h, img_w = img.shape[:2]
        annotations = []
        for anno in trainA["annotations"]:
            if anno["image_id"] == image["id"]:
                x, y, w, h = anno["bbox"]
                # annotations.append((x, y, w, h))
                annotations.append((x, y, x + w, y + h))
                # cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0))
        t = transforms(image=img, bboxes=annotations)
        img_trans, bbox_trans = t["image"], t["bboxes"]
        # print(annotations)
        # print(bbox_trans)
        # exit()
        bbox_trans = [((x1 + img_w) % img_w, (y1 + img_h) % img_h, (x2 + img_w) % img_w, (y2 + img_h) % img_h) for
                      x1, y1, x2, y2 in bbox_trans]
        print(annotations)
        print(bbox_trans)

        for anno, anno_trans in zip(annotations, bbox_trans):
            # cv2.rectangle(img, pt1=(anno[0], anno[1]), pt2=(anno[0] + anno[2], anno[1] + anno[3]), color=(255, 0, 0))
            # cv2.rectangle(img_trans, pt1=(anno_trans[0], anno_trans[1]),
            #               pt2=(anno_trans[0] + anno_trans[2], anno_trans[1] + anno_trans[3]), color=(255, 0, 0))
            cv2.rectangle(img, pt1=(anno[0], anno[1]), pt2=(anno[2], anno[3]), color=(255, 0, 0))
            cv2.rectangle(img_trans, pt1=(anno_trans[0], anno_trans[1]),
                          pt2=(anno_trans[2], anno_trans[3]), color=(255, 0, 0))
        cv2.imshow("img", img)
        cv2.imshow("img_trans", img_trans)
        cv2.waitKey()
