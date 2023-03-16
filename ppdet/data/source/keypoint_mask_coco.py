"""
@File  : keypoint_mask_coco.py
@Author: tao.jing
@Date  : 2022/12/20
@Desc  :
"""
from ppdet.data.source.keypoint_coco import KeypointTopDownCocoDataset
import os
from pathlib import Path
import cv2
import numpy as np
import json
import copy
import pycocotools
from pycocotools.coco import COCO
from .dataset import DetDataset
from ppdet.core.workspace import register, serializable


@register
@serializable
class KeypointTopDownCocoMaskDataset(KeypointTopDownCocoDataset):
    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 trainsize,
                 transform=[],
                 bbox_file=None,
                 use_gt_bbox=True,
                 pixel_std=200,
                 image_thre=0.0):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints, trainsize,
                         transform, bbox_file, use_gt_bbox, pixel_std, image_thre)

    def __getitem__(self, idx):
        """Prepare sample for training given the index."""
        records = copy.deepcopy(self.db[idx])
        img_path = records['image_file']
        mask_path = img_path.replace('.jpg', '.png')
        records['image'] = cv2.imread(mask_path, cv2.IMREAD_COLOR |
                                      cv2.IMREAD_IGNORE_ORIENTATION)
        records['image'] = cv2.cvtColor(records['image'], cv2.COLOR_BGR2RGB)
        records['score'] = records['score'] if 'score' in records else 1
        records = self.transform(records)
        # print('records', records)
        return records

