"""
@File  : kp_transform_test.py
@Author: tao.jing
@Date  : 2022/4/3
@Desc  :
"""
import numpy as np
import matplotlib.pyplot as plt

import cv2

from ppdet.data.transform.keypoint_operators import ToHeatmaps
from ppdet.data.transform.keypoint_operators import ToHeatmapsTopDown

from ppdet.smoke_kp.skp_utils import parse_annot
from ppdet import smoke_kp as SKT


def generate_smoke_kp_hm():
    data_rec = 'D:\Dataset\smoke\HR\labelled\images\S000000.jpg  1245.38   428.15     0.54   526.85'
    data_eles = parse_annot(data_rec)

    img_path = data_eles['img_path']
    ori_kp_coors = data_eles['kp_coors']

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, kp_coors = SKT.Resize(256, 64) \
        (img, ori_kp_coors)

    coco_coors = np.zeros((2, 3))
    coco_coors[:, :2] = kp_coors
    coco_coors[:, 2] = np.asarray([2, 2])

    num_joints = 2
    hm_sizes = [64, ]
    sigma = 2
    hm_generator = ToHeatmaps(num_joints, hm_sizes, sigma)
    # 0 - hm scales index; 1 - people index
    coco_coors = coco_coors[np.newaxis, :]
    mask = np.zeros((64, 64))
    records = {
        'joints': [coco_coors, ],
        'mask': [mask, ],
    }
    records = hm_generator(records)
    hm = records['heatmap_gt1x']

    plt.matshow(hm[0])
    plt.show()
    plt.matshow(hm[1])
    plt.show()

    merged_hm = hm[0] + hm[1]
    merged_hm = cv2.resize(merged_hm, (img.shape[0], img.shape[1]))

    plt.imshow(img)
    plt.imshow(merged_hm, alpha=0.5, interpolation='nearest', cmap='jet')
    plt.show()


def generate_smoke_kp_hm_topdown():
    num_joints = 2
    # hmsize = [256 // 4, 256 // 4]
    hmsize = [1280, ]
    sigma = 2
    hm_generator = ToHeatmapsTopDown(hmsize, sigma)
    kp_coors = np.asarray([[[1245.38, 428.15, 0], [0.54, 526.85, 0]], ])
    kp_coors_vis = np.asarray([[1, 1, 2], [1, 1, 2]])
    img = np.ones((1280, 1280))
    records = {
        'joints': kp_coors,
        'joints_vis': kp_coors_vis,
        'mask': [0, 0],
        'image': img
    }
    records = hm_generator(records)
    hm = records['target']

    plt.matshow(hm[0])
    plt.show()
    plt.matshow(hm[1])
    plt.show()



if __name__ == '__main__':
    generate_smoke_kp_hm()
    #generate_smoke_kp_hm_topdown()