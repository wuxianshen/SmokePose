"""
@File  : skp_trans_ops.py
@Author: tao.jing
@Date  : 2022/5/6
@Desc  :
"""
from paddle.vision.transforms import functional as F
from ppdet.core.workspace import serializable

import numpy as np


__all__ = [
    'SKPResize'
]


def register_keypointop(cls):
    return serializable(cls)

@register_keypointop
class SKPResize(object):
    def __init__(self, trainsize):
        if not isinstance(trainsize, list):
            raise RuntimeError(f'Invalid trainsize {trainsize}')
        self.trainsize = trainsize

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints = np.squeeze(joints)

        im_h, im_w, _ = image.shape
        image = F.resize(image, self.trainsize)
        for joint in joints:
            feat_stride_x = im_w / self.trainsize[0]
            feat_stride_y = im_h / self.trainsize[1]
            mu_x = int(joint[0] / feat_stride_x + 0.5)
            mu_y = int(joint[1] / feat_stride_y + 0.5)
            joint[0] = mu_x
            joint[1] = mu_y
        records['image'] = image
        records['joints'] = joints
        return records
