"""
@File  : data_config.py
@Author: tao.jing
@Date  : 2022/4/7
@Desc  :
"""
from ppdet.task_config.root_config import _C, CN

_C.DATASET = CN()

# Dataset hyper-parameters
# SKP, COCO, MPII
_C.DATASET.DATA_TYPE = 'SKP'
_C.DATASET.SKP_OKS_SIGMAS = (1.0, )