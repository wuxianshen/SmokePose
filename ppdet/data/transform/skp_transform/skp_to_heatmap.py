"""
@File  : skp_to_heatmap.py
@Author: tao.jing
@Date  : 2022/5/13
@Desc  :
"""
import numpy as np

import paddle

from ppdet.core.workspace import serializable


__all__ = [
    'SKPToHeatmaps'
]


def register_keypointop(cls):
    return serializable(cls)


@register_keypointop
class SKPToHeatmaps(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        num_joints (int): the keypoint numbers of dataset to train
        hmsize (list[2]): output heatmap's shape list of different scale outputs of higherhrnet
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image, mask and coords

    Returns:
        records(dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self,
                 num_joints,
                 hmsize,
                 target_idx = 0,
                 sigma=None):
        super(SKPToHeatmaps, self).__init__()

        self.num_joints = num_joints
        self.hmsize = np.array(hmsize)
        self.target_idx = target_idx
        if sigma is None:
            sigma = hmsize[0] // 64
        self.sigma = sigma

        r = 6 * sigma + 3
        x = np.arange(0, r, 1, np.float32)
        y = x[:, None]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, records):
        kpts_lst = records['joints']
        mask_lst = records['mask']
        for idx, hmsize in enumerate(self.hmsize):
            mask = mask_lst[idx]
            kpts = kpts_lst[idx]
            heatmaps = np.zeros((self.num_joints, hmsize, hmsize))
            inds = np.where(kpts[..., 2] > 0)
            visible = kpts[inds].astype(np.int64)[..., :2]
            ul = np.round(visible - 3 * self.sigma - 1)
            br = np.round(visible + 3 * self.sigma + 2)
            sul = np.maximum(0, -ul)
            sbr = np.minimum(hmsize, br) - ul
            dul = np.clip(ul, 0, hmsize - 1)
            dbr = np.clip(br, 0, hmsize)
            for i in range(len(visible)):
                if visible[i][0] < 0 or visible[i][1] < 0 or visible[i][
                        0] >= hmsize or visible[i][1] >= hmsize:
                    continue
                dx1, dy1 = dul[i]
                dx2, dy2 = dbr[i]
                sx1, sy1 = sul[i]
                sx2, sy2 = sbr[i]
                heatmaps[inds[1][i], dy1:dy2, dx1:dx2] = np.maximum(
                    self.gaussian[sy1:sy2, sx1:sx2],
                    heatmaps[inds[1][i], dy1:dy2, dx1:dx2])
            records['heatmap_gt{}x'.format(idx + 1)] = heatmaps
            records['mask_{}x'.format(idx + 1)] = mask

            if self.target_idx == idx:
                records['target'] = heatmaps.astype(np.float32)
                target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
                records['target_weight'] = target_weight

        del records['mask']
        return records