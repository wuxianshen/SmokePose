"""
@File  : model_preview.py
@Author: tao.jing
@Date  : 2022/3/14
@Desc  :
"""
import argparse
import os

import paddle
from ppdet.modeling import VGG

from ppdet.task_config import _C, update_config


def show_model(network):
    model = paddle.Model(network)
    model.summary(input_size=(1, 3, 512, 512))


def model_preview():
    network = VGG()
    show_model(network)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual Transformer for semantic segmentation')
    parser.add_argument(
        "--config",
        dest='task_cfg',
        default=None,
        type=str,
        help="The config file."
    )
    return parser.parse_args()


if __name__ == '__main__':
    #model_preview()
    args = parse_args()
    args.task_cfg = '../local_data/task_yml/skp.yml'
    update_config(_C, args)

    print(_C.DATASET.DATA_TYPE)
