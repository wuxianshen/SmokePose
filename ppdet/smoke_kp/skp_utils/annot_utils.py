"""
@File  : annot_utils.py
@Author: tao.jing
@Date  : 2022/2/23
@Desc  :
"""

import numpy as np

__all__ = [
    'parse_annot',
    'kp_coors_verify'
]


def parse_annot(data_desc: str) -> dict:
    """
    # data_desc: arg0 arg1 arg2 arg3 arg4
    # arg0: Image absolute path
    # (arg1, arg2): Source point (X, Y)
    # (arg3, arg4): Vanish point (X, Y)

    Parse data description to data elements.
    :param data_desc: str
    :return: dict
    """
    data_eles = data_desc.strip().split()
    assert len(data_eles) == 5, \
        f'[SmokeAnalysisDataset] Invalid data: {data_eles}'

    img_path = data_eles[0]
    source_point = [float(data_eles[1]), float(data_eles[2])]
    vanish_point = [float(data_eles[3]), float(data_eles[4])]
    kp_coors = np.array([source_point, vanish_point])
    return {
        'img_path': img_path,
        'kp_coors': kp_coors
    }


def kp_coors_verify(kp_coors: np.ndarray) -> np.ndarray:
    """
    Verify kp_coors.
    :param kp_coors:
    :return:
    """
    kp_shape = kp_coors.shape
    if len(kp_shape) == 1:
        assert len(kp_coors) == 2, \
            f'Invalid kp_coors shape {kp_coors}'
        kp_coors = kp_coors[np.newaxis, :]
    elif len(kp_shape) > 2:
        assert False, \
            f'Invalid kp_coors {kp_coors}'
    return kp_coors
