"""
@File  : skp_multi_hm_generator.py
@Author: tao.jing
@Date  : 2022/5/23
@Desc  :
"""
from typing import Optional, Union

from ppdet.core.workspace import serializable
from ppdet.skp_utils import img_size_param_check


__all__ = [
    'SKPMultiHmGenerator'
]


def register_keypointop(cls):
    return serializable(cls)


@register_keypointop
class SKPMultiHmGenerator(object):
    def __init__(self,
                 kp_num=2,
                 img_size: Optional[Union[int, float, tuple, list]]=(512, 512),
                 hm_stride_list: Optional[Union[tuple, list]]=(4, 8)):
        self.kp_num = kp_num
        self.img_size = img_size_param_check(img_size)
        self.hm_stride_list = hm_stride_list

    def __call__(self, records):
        for idx, hm_stride in enumerate(self.hm_stride_list):
            hm_name = f'heatmap_gt{idx + 1}x'
            if hm_name not in records:
                raise ValueError(f'[{self.__class__}] {hm_name} not in records.')
            hm = records[f'heatmap_gt{idx + 1}x']
            kp_num, hm_w, hm_h = hm.shape

            if (hm_w != int(self.img_size[0] / hm_stride)) or \
                (hm_h != int(self.img_size[1] / hm_stride)):
                raise ValueError(f'Not matched stride: Index {idx} '
                                 f'stride: {hm_stride} '
                                 f'img_size {self.img_size} '
                                 f'hm_size {hm_w} {hm_h}')
            records[f'hm_s{hm_stride}'] = hm.copy()
        return records





