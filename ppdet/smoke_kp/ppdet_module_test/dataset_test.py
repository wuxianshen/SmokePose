"""
@File  : dataset_test.py
@Author: tao.jing
@Date  : 2022/4/4
@Desc  :
"""
import os
from ppdet.data import KeypointBottomUpCocoDataset


def bottom_up_dataset_test():
    dataset_dir = 'E:\\Dataset\\COCO\\'
    image_dir = os.path.join(dataset_dir, 'val2017')
    anno_path = os.path.join(dataset_dir,
                             'annotations', 'person_keypoints_val2017.json')
    num_joints = 17
    transform = [],
    shard = [0, 1],
    test_mode = True
    kp_bu_dataset = KeypointBottomUpCocoDataset()



if __name__ == '__main__':
    bottom_up_dataset_test()