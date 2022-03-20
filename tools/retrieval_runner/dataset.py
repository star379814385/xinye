from torch.utils.data import Dataset, DataLoader
import os
import json
# from lrh_utils.utils.cv2_utils import cv_imread
import cv2
import numpy as np
from tqdm import tqdm
import torch
from albumentations import *
from copy import deepcopy


#
class PadResize(DualTransform):

    def __init__(self, size, const_value=(0, 0, 0), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(PadResize, self).__init__(always_apply, p)
        self.size = size
        self.interpolation = interpolation
        self.background = np.array([[const_value]], dtype=np.uint8).repeat(self.size, 0).repeat(self.size, 1)

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        target_size = self.size
        interpolation = self.interpolation
        w, h = img.shape[:2]
        img_xx = cv2.resize(img, (self.size, self.size))
        if w > h:
            img = cv2.resize(img, (target_size, int(h * target_size * 1.0 / w)), interpolation=interpolation)
        else:
            img = cv2.resize(img, (int(w * target_size * 1.0 / h), int(target_size)), interpolation=interpolation)

        ret_img = deepcopy(self.background)
        h, w = img.shape[:2]
        st_h = int((ret_img.shape[0] - h) / 2.0)
        st_w = int((ret_img.shape[1] - w) / 2.0)
        ret_img[st_h: st_h + h, st_w: st_w + w] = img

        # ret_img = np.hstack([ret_img, img_xx])
        return ret_img

        # return F.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")


class QuiryDataset(Dataset):
    def __init__(self, root, height, width, phase="train"):
        super(QuiryDataset, self).__init__()
        self.root = root
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.width, self.height = width, height
        # init
        self.make_dataset()
        self.transforms_cv2_train = self.get_transform_train()
        self.transforms_cv2_test = self.get_transform_test()
        # self.padresize = PadResize((224, 244))

    def make_dataset(self):
        self.samples = []
        for root, path, files in os.walk(self.root):
            for file in files:
                image_name = os.path.join(root, file).split(self.root)[-1]
                cls_idx = int(image_name.split("/")[1].split("_")[0])
                # cls_idx = int(image_name.split("\\")[1].split("_")[0])
                self.samples.append((image_name, cls_idx))
        print("loading {} samples from{}".format(len(self.samples), self.root))

    def __getitem__(self, i):
        """

        :param sample:
        :return: image [np.uint8, (c, h, w)], pixel [np.float32, (c, h, w)], label [int]
        """
        image_name, label = self.samples[i]
        image = cv2.imread(self.root + image_name)
        # trans bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.phase == "train":
            image = self.transforms_cv2_train(image=image)["image"]
        elif self.phase == "val":
            image = self.transforms_cv2_test(image=image)["image"]
        else:
            # for test
            pass

        return image, label

    def get_transform_train(self):
        transforms = Compose([
            # RandomRotate90(p=0.5),
            # 仿射变换相关
            OneOf([
                HorizontalFlip(p=1),
                VerticalFlip(p=1),
                Compose([HorizontalFlip(p=1), VerticalFlip(p=1)], p=1)
            ], p=0.75),
            # 裁剪
            OneOf([
                # Compose([Resize(int(self.width * 1.5), int(self.height * 1.5), p=1),
                #          RandomCrop(self.width, self.height, p=1)], p=1),
                PadResize(self.width, p=1),
                # Resize(self.width, self.height, p=1),
            ], p=1),

            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-45, 45), p=1),
            # 灰度锐化或模糊
            CLAHE(p=0.3),
            # OneOf([
            #     MotionBlur(blur_limit=(3, 5), p=1),
            #     Blur(blur_limit=5, p=1),
            #     MedianBlur(blur_limit=5, p=1)
            # ], p=0.5),
            # 灰度抖动
            # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
            GaussNoise(var_limit=(10, 30), mean=0, p=0.5),
            # norm, to float
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            # to tensor
        ], p=1.0)
        return transforms

    def get_transform_test(self):
        transforms = Compose([
            PadResize(self.width, p=1),
            # Resize(self.width, self.height, p=1),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ])
        return transforms

    @staticmethod
    def collate_fn(data):
        # collect and to tensor
        image_tensor = torch.tensor(np.stack([dat[0] for dat in data]).transpose(0, 3, 1, 2), dtype=torch.float32)
        label_tensor = torch.tensor([dat[1] for dat in data], dtype=torch.long)
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.samples)


class KeyDataset(Dataset):
    def __init__(self, root, height, width, use_augs=False, **kwargs):
        super(KeyDataset, self).__init__()
        self.root = root
        self.width, self.height = width, height
        self.make_dataset()
        self.transforms_cv2_test = self.get_transform_test()
        self.key_augs = self.get_key_aug() if use_augs else None
        if self.key_augs is None:
            self.collate_fn = KeyDataset.collate_fn1
        else:
            self.collate_fn = KeyDataset.collate_fn2

    def make_dataset(self):
        self.samples = []
        for root, path, files in os.walk(self.root):
            for file in files:
                image_name = os.path.join(root, file).split(self.root)[-1]
                cls_idx = int(image_name.split("/")[1].split("_")[0])
                # cls_idx = int(image_name.split("\\")[1].split("_")[0])
                self.samples.append((image_name, cls_idx))
        print("loading {} samples from{}".format(len(self.samples), self.root))

    def __getitem__(self, i):
        image_name, label = self.samples[i]
        image = cv2.imread(self.root + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.key_augs is None:
            image = self.transforms_cv2_test(image=image)["image"]
            return image, label
        else:
            images = [image] + [aug(image=image)["image"] for aug in self.key_augs]
            images = np.stack([self.transforms_cv2_test(image=image)["image"] for image in images])
            labels = np.array([label] * images.shape[0])
            return images, labels

    def get_transform_test(self):
        transforms = Compose([
            PadResize(self.width, p=1),
            # Resize(self.width, self.height, p=1),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ])
        return transforms

    def get_key_aug(self):
        aug_list = [
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            Compose([HorizontalFlip(p=1), VerticalFlip(p=1)]),
        ]
        # aug_all1 = ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-45, 45), p=1)
        # aug_list = aug_list + [aug_all1] + [Compose([aug, aug_all1]) for aug in aug_list]
        return aug_list

    @staticmethod
    def collate_fn1(data):
        # collect and to tensor
        image_tensor = torch.tensor(np.stack([dat[0] for dat in data]).transpose(0, 3, 1, 2), dtype=torch.float32)
        label_tensor = torch.tensor([dat[1] for dat in data], dtype=torch.long)
        # print(image_tensor.shape)
        # print(label_tensor.shape)
        return image_tensor, label_tensor

    @staticmethod
    def collate_fn2(data):
        # collect and to tensor
        image_tensor = torch.tensor(np.concatenate([dat[0] for dat in data]).transpose(0, 3, 1, 2), dtype=torch.float32)
        label_tensor = torch.tensor(np.concatenate([dat[1] for dat in data]), dtype=torch.long)
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    quiry_data = QuiryDataset(
        root="../data/初赛数据集/train/a_images_crop",
        phase="train",
        width=256, height=256,
    )
    key_data = KeyDataset(
        root="../data/初赛数据集/train/b_images_crop",
        phase="val",
        width=256, height=256,
    )
    print(len(quiry_data))
