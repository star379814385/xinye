"""
    this is transform by cv2 image
    input_type : np.uint8
"""
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

"""
add albumentations lib 
"""
# from albumentations import *



class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms
        assert isinstance(self.transforms, list)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class BaseTransform(object):
    def __init__(self, p=1, group_transform=False):
        self.p = p
        self.group_transform = group_transform

    def __call__(self, x):
        """
        :param x: if gropu_transform is True, x must be a image list, else x must be a image.
        :return: return type as x
        """
        if np.random.random() > self.p:
            return x
        return [self.fun(i) for i in x] if self.group_transform else self.fun(x)

    def fun(self, x):
        raise


class RandomFlip(BaseTransform):
    def __init__(self, p=0.5, horizontal=True, group_transform=False):
        super(RandomFlip, self).__init__(p=p, group_transform=group_transform)
        self.horizontal = horizontal

    def fun(self, x):
        return self.horizontal_filp(x) if self.horizontal else self.vertical_flip(x)

    def horizontal_filp(self, x):
        if x.ndim == 3:
            x = x[:, ::-1, :]
        elif x.ndim == 2:
            x = x[:, ::-1]
        else:
            raise
        return x

    def vertical_flip(self, x):
        if x.ndim == 3:
            x = x[::-1, :, :]
        elif x.ndim == 2:
            x = x[::-1, :]
        else:
            raise
        return x


class Resize(BaseTransform):
    def __init__(self, width, height, group_transform=False):
        super(Resize, self).__init__(group_transform=group_transform)
        self.width, self.height = width, height

    def fun(self, x):
        x = cv2.resize(x, dsize=(self.width, self.height))
        return x


class RandomCircleConcat(BaseTransform):
    def __init__(self, horizontal=False, group_transform=False):
        super(RandomCircleConcat, self).__init__(group_transform=group_transform)
        self.horizontal = horizontal
        self.rand = 0

    def __call__(self, x):
        """
        :param x: if gropu_transform is True, x must be a image list, else x must be a image.
        :return: return type as x
        """
        self.rand = np.random.random()
        return [self.fun(i) for i in x] if self.group_transform else self.fun(x)

    def fun(self, x):
        return self.horizontal_circle_concat(x) if self.horizontal else self.vertical_circle_concat(x)

    def horizontal_circle_concat(self, x):
        width = x.shape[1]
        concat_point = int(self.rand * width)
        if x.ndim == 3:
            return np.concatenate([x[:, concat_point:, :], x[:, :concat_point, :]], axis=1)
        elif x.ndim == 2:
            return np.concatenate([x[:, concat_point:], x[:, :concat_point]], axis=1)
        else:
            raise

    def vertical_circle_concat(self, x):
        height = x.shape[0]
        concat_point = int(self.rand * height)
        if x.ndim == 3:
            return np.concatenate([x[concat_point:, :, :], x[:concat_point, :, :]], axis=0)
        elif x.ndim == 2:
            return np.concatenate([x[concat_point:, :], x[:concat_point, :]], axis=0)
        else:
            raise


class ColorJitterGray(object):
    def __init__(self, brightness=0, contrast=0):
        super(ColorJitterGray, self).__init__()
        self.brightness = brightness
        self.contrast = contrast

    def adjust_brightness(self, x):
        brightness_jit = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        x = x.astype(np.float32)
        x = x * brightness_jit
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def adjust_contrast(self, x):
        contrast_jit = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        Mean = np.mean(x)
        x = x.astype(np.float32)
        x = x * contrast_jit + Mean * (1 - contrast_jit)
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def __call__(self, x):
        x = self.adjust_brightness(x)
        x = self.adjust_contrast(x)
        return x


class ElasticDistort(object):
    def __init__(self, alphaX=5, alphaY=5, sigmaX=15, sigmaY=15, alpha_jit=False, random_state=None, group_transform=False):
        super(ElasticDistort, self).__init__()
        self.alphaX, self.alphaY = alphaX, alphaY
        self.sigmaX, self.sigmaY = sigmaX, sigmaY
        self.alpha_jit = alpha_jit
        self.group_transform = group_transform
        self.random_state = random_state

    def __call__(self, x):
        if isinstance(x, list):
            height, width = x[0].shape[:2]
        else:
            height, width = x.shape[:2]
        alphaX = np.abs(np.random.uniform(-self.alphaX, self.alphaX)) if self.alpha_jit else self.alphaX
        alphaY = np.abs(np.random.uniform(-self.alphaY, self.alphaY)) if self.alpha_jit else self.alphaY
        sigmaX = self.sigmaX
        sigmaY = self.alphaY
        indices = self.get_elastic_transform2d_indices(width, height, alphaX, alphaY, sigmaX, sigmaY, self.random_state)
        if self.group_transform:
            return [map_coordinates(i, indices, order=1).reshape(i.shape) for i in x]
        else:
            return map_coordinates(x, indices, order=1).reshape(x.shape)

    @staticmethod
    def get_elastic_transform2d_indices(width, height, alphaX, alphaY, sigmaX, sigmaY, random_state=None):
        """
        obtained from: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
        edited by lrh 2020.12.18
        """

        if random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state = np.random.RandomState(random_state)

        shape = height, width
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmaX, mode="constant", cval=0)
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmaY, mode="constant", cval=0)
        dx = (dx - np.min(dx)) / np.max(dx) - np.min(dx)
        dy = (dy - np.min(dy)) / np.max(dy) - np.min(dy)
        dx = dx * alphaX
        dy = dy * alphaY

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(y + dy, (1, -1)), np.reshape(x + dx, (1, -1))

        return indices

    @staticmethod
    def elastic_transform2d(image, alphaX, alphaY, sigmaX, sigmaY, random_state=None):
        height, width = image.shape[:2]
        indices = ElasticDistort.get_elastic_transform2d_indices(width, height, alphaX, alphaY, sigmaX, sigmaY,
                                                                 random_state)

        return map_coordinates(image, indices, order=1).reshape(image.shape)


if __name__ == "__main__":
    img = cv2.imread("000_1_4.jpg", -1)
    transform = ComposeJoint(transforms=[
        Resize(width=256, height=256),
        RandomFlip(p=0.5, horizontal=True, group_transform=False),
        RandomFlip(p=0.5, horizontal=False, group_transform=False),
        RandomCircleConcat(p=0.5, horizontal=True, group_transform=False)
    ])
    img_trans = transform(img)
    cv2.imshow("img", img)
    cv2.imshow("img_trans", img_trans)
    cv2.waitKey()
