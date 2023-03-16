"""
@File  : skp_dataset.py
@Author: tao.jing
@Date  : 2022/4/5
@Desc  :
"""
import os

import numpy as np

from ppdet.core.workspace import register, serializable
from ppdet.data import KeypointBottomUpBaseDataset
from ppdet.data import Compose
from ppdet.smoke_kp.skp_data.skp_datasets.skp_json_parser import SKPJsonParser


__all__ = [
    'SKPDataset'
]

@register
@serializable
class SKPDataset(KeypointBottomUpBaseDataset):
    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 train_size=(256, 256),
                 transform=[],
                 shard=[0, 1],
                 test_mode=False,
                 load_ratio=1.0):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform, shard, test_mode)

        self.ann_file = os.path.join(dataset_dir, anno_path)
        self.train_size = train_size
        self.pixel_std = 200
        self.shard = shard
        self.test_mode = test_mode

        self.load_ratio = load_ratio

    def parse_dataset(self):
        self.skp_db = SKPJsonParser(self.ann_file, self.load_ratio)

    def __len__(self):
        """Get dataset length."""
        return len(self.skp_db)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = w * 1.0 / h

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _get_imganno(self, idx):
        """Get anno for a single image.

        Args:
            idx (int): image idx

        Returns:
            dict: info for model training
        """
        data_des = self.skp_db[idx]
        img_name = data_des['img_name']
        im_shape = np.asarray([data_des['img_height'], data_des['img_width']])

        annots = data_des['annotations']
        assert len(annots) == 1, \
            f'Invalid annotations len {len(annots)}'

        keypoints = np.asarray(annots[0]['keypoints'])
        assert len(keypoints) == 4, \
            f'Invalid keypoints {keypoints}'

        keypoints = keypoints.reshape(2, 2)
        keypoints = np.c_[keypoints, np.asarray([[2,], [2,]])]

        #keypoints = keypoints[np.newaxis, :] # TopDown No this line
        joints = keypoints
        joints_vis = np.ones(tuple(joints.shape))

        center, scale = self._box2cs((0, 0, data_des['img_width'], data_des['img_height']))

        db_rec = {}
        db_rec['image_file'] = os.path.join(self.img_prefix, img_name)
        db_rec['mask'] = np.ones(tuple(im_shape.shape))
        db_rec['joints'] = joints
        db_rec['im_shape'] = im_shape
        db_rec['joints_vis'] = joints_vis
        db_rec['center'] = center
        db_rec['scale'] = scale
        db_rec['score'] = 1
        db_rec['im_id'] = idx

        return db_rec


if __name__ == '__main__':
    dataset_dir = 'D:\\Dataset\\smoke_datafolder\\base\\'
    image_dir = 'images'
    anno_path = 'annotations\\smoke_keypoint.json'
    num_joints = 2
    transform = Compose([{'ToHeatmaps': {'num_joints': 2, 'hmsize': [64, ], 'sigma': 2}}, ])

    skp_dataset = SKPDataset(dataset_dir=dataset_dir,
                             image_dir=image_dir,
                             anno_path=anno_path,
                             num_joints=num_joints,
                             transform=transform)

    skp_dataset.parse_dataset()
    db_rec = skp_dataset[0]
    print(db_rec)