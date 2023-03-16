"""
@File  : config_entry.py
@Author: tao.jing
@Date  : 2022/4/7
@Desc  :
"""
import os
import yaml

from ppdet.task_config.root_config import _C
from ppdet.task_config.dataset_config import *

__all__ = ['get_config', '_C', 'update_config']


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('merging config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    """Update config by ArgumentParser
    Args:
        args: ArgumentParser contains options
    Return:
        config: updated config
    """
    if args.task_cfg:
        _update_config_from_file(config, args.task_cfg)
    config.defrost()
    return config


def get_config():
    config = _C.clone()
    return config


if __name__ == '__main__':
    pass