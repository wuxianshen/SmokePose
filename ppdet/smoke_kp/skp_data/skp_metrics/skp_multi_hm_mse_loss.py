"""
@File  : skp_multi_hm_mse_loss.py
@Author: tao.jing
@Date  : 2022/5/22
@Desc  :
"""
import paddle
from paddle import nn

from ppdet.core.workspace import register, serializable

__all__ = [
    'MultiHmMSELoss'
]


@register
@serializable
class MultiHmMSELoss(nn.Layer):
    def __init__(self,
                 hm_stride_list=[4, 8],
                 hm_weights=[0.7, 0.3]):
        '''
        Require hm_sx in records.
        '''
        super(MultiHmMSELoss, self).__init__()
        self.hm_stride_list = hm_stride_list
        self.hm_weights = hm_weights
        self.criterion = nn.MSELoss()

        if len(hm_stride_list) != len(hm_weights):
            assert ValueError(f'[MultiHmMSELoss] hm_stride_list '
                              f'and hm_weights have different length.')

    def forward(self, output, records):
        if len(output) != len(self.hm_stride_list):
            raise ValueError(f'[MultiHmMSELoss] '
                             f'Invalid output len {len(output)} '
                             f'for hm_list {self.hm_stride_list}')

        losses = list()
        for hm_idx, hm_stride in enumerate(self.hm_stride_list):
            target = records[f'hm_s{hm_stride}'].cast('float32')
            predict = output[hm_idx].cast('float32')
            # batch_size, kp_num, H, W = predict.shape
            losses.append(self.criterion(predict, target))

        # loss = paddle.mean(paddle.stack(losses, axis=0))
        loss = paddle.zeros((1,))
        for idx, hm_weight in enumerate(self.hm_weights):
            loss += losses[idx] * hm_weight

        keypoint_losses = {
            'loss': loss
        }
        return keypoint_losses