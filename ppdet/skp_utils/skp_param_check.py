"""
@File  : skp_param_check.py
@Author: tao.jing
@Date  : 2022/5/24
@Desc  :
"""
from typing import Union

__all__ = [
    'img_size_param_check'
]


def img_size_param_check(img_size: Union[int, float, tuple, list]) \
        -> Union[list, tuple]:
    if isinstance(img_size, float):
        img_size = int(img_size)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    if not (isinstance(img_size, tuple) or
        isinstance(img_size, list)):
        raise ValueError(f'Invalid img_size type {img_size}')
    if len(img_size) != 2:
        raise ValueError(f'Invalid img_size len {img_size}')
    return img_size
