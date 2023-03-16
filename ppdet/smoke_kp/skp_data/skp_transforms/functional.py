"""
@File  : functional.py
@Author: tao.jing
@Date  : 2022/2/4
@Desc  :
"""

import numpy as np

from PIL import Image


def np_input_decorator(func):
    def wrapper(im, magnitude, **kwargs):
        input_ndarray = False
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im.astype(np.uint8))
            input_ndarray = True
        im = func(im, magnitude, **kwargs)
        if input_ndarray:
            return np.asarray(im)
        return im
    return wrapper


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


if __name__ == '__main__':
    pass