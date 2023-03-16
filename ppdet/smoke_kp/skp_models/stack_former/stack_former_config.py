"""
@File  : stack_former_config.py
@Author: tao.jing
@Date  : 2022/5/29
@Desc  :
"""


StackFormer_B0 = {
    'down_former_chans' : [32, 64, 160, 256],
    'up_former_chans' : [128, 128, 64],
    'up_num_heads': [4, 2, 1],
    'up_mlp_ratios': [2, 2, 2]
}


StackFormer_B1 = {
    'down_former_chans' : [64, 128, 320, 512],
    'up_former_chans' : [320, 256, 128],
    'up_num_heads': [4, 2, 1],
    'up_mlp_ratios': [2, 2, 2]
}
