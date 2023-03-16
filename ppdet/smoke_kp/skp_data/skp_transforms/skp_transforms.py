"""
@File  : kp_transforms.py
@Author: tao.jing
@Date  : 2022/1/23
@Desc  :
"""
from collections.abc import Sequence
import numpy as np

from paddle.vision.transforms import functional as F
from .functional import normalize

__all__ = [
    'Compose',
    'Resize',
    'Normalize'
]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im, targets=None):
        for t in self.transforms:
            if targets is not None:
                im, targets = t(im, targets)
            else:
                im = t(im)
        if targets is not None:
            return (im, targets)
        else:
            return (im, )


class Resize(object):
    def __init__(self, img_size, hm_size):
        """
        :param img_size: (W, H) -- (x, y)
        :param hm_size: (W, H) -- (x, y)
        """
        if not isinstance(img_size, (int, Sequence)):
            raise TypeError(f'Image size should be int or sequence. Got {type(img_size)}')
        if isinstance(img_size, Sequence) and len(img_size) not in (1, 2):
            raise ValueError('If image size is a sequence, it should have 1 or 2 values')
        if not isinstance(hm_size, (int, Sequence)):
            raise TypeError(f'Heatmap size should be int or sequence. Got {type(hm_size)}')
        if isinstance(hm_size, Sequence) and len(hm_size) not in (1, 2):
            raise ValueError('If heatmap size is a sequence, it should have 1 or 2 values')

        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        self.img_size_list = img_size

        if isinstance(hm_size, int):
            hm_size = [hm_size, hm_size]
        self.hm_size_list = hm_size

    def __call__(self, im, kp_coors=None):
        """
        Resize image and the corresponding kp_coors.
        Adjust kp_coors according to image size, not hm_size.
        :param im: Image, ndarray
        :param kp_coors: KeyPoint coordinates, ndarray, shape: (kp_num, 2)
        :return: (im_resized, kp_coors_resized)
        """
        im_h, im_w, _ = im.shape
        im_resized = F.resize(im, self.img_size_list)
        if kp_coors is not None:
            kp_coors_resized = np.zeros(kp_coors.shape)
            kp_num = kp_coors.shape[0]
            # Adjust by image size before and after resize
            feat_stride_x = im_w / self.hm_size_list[0]
            feat_stride_y = im_h / self.hm_size_list[1]
            for kp_idx in range(kp_num):
                mu_x = int(kp_coors[kp_idx][0] / feat_stride_x + 0.5)
                mu_y = int(kp_coors[kp_idx][1] / feat_stride_y + 0.5)
                kp_coors_resized[kp_idx][0] = mu_x
                kp_coors_resized[kp_idx][1] = mu_y
            return (im_resized, kp_coors_resized)
        return (im_resized, )


class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, kp_coors=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if kp_coors is None:
            return (im, )
        else:
            return (im, kp_coors)


if __name__ == '__main__':
    pass
