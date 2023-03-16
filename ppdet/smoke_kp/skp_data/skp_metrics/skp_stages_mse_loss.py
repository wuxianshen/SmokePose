"""
@File  : skp_stages_mse_loss.py
@Author: tao.jing
@Date  : 2022/5/10
@Desc  :
"""

import paddle
from paddle import nn

from ppdet.core.workspace import register, serializable

__all__ = [
    'SingleStageMSELoss',
    'MultiStagesMSELoss'
]


@register
@serializable
class SingleStageMSELoss(nn.Layer):
    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.0):
        super(SingleStageMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, out, target, target_weight):
        assert len(out.shape) == 4, \
            f'Invalid out shape {out.shape}'
        B, kp_num = out.shape[0], out.shape[1]

        hm_preds = out.reshape((B, kp_num, -1)).split(kp_num, 1)
        hm_gts = target.reshape((B, kp_num, -1)).split(kp_num, 1)

        losses = list()
        for kp_idx in range(kp_num):
            hm_pred = hm_preds[kp_idx].squeeze(1).astype('float32')
            hm_gt = hm_gts[kp_idx].squeeze(1).astype('float32')
            if self.use_target_weight:
                losses.append(self.criterion(hm_pred * target_weight[:, kp_idx],
                                       hm_gt * target_weight[:, kp_idx]))
            else:
                losses.append(self.criterion(hm_pred, hm_gt))
        return paddle.mean(paddle.stack(losses, axis=0)) * self.loss_weight


@register
@serializable
class MultiStagesMSELoss(nn.Layer):
    def __init__(self,
                 out_stage_num=8,
                 use_target_weight=False,
                 loss_weight=1.0):
        super(MultiStagesMSELoss, self).__init__()
        self.criterion = SingleStageMSELoss()
        self.out_stage_num = out_stage_num
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, records):
        target = records['target']
        target_weight = records['target_weight']

        batch_size, out_stage_num, kp_num, H, W = output.shape
        assert out_stage_num == self.out_stage_num, \
            f'Out stage num not matched {self.out_stage_num} / {out_stage_num}'

        out_stages = output.reshape(
            (batch_size, out_stage_num, -1)).split(out_stage_num, 1)

        losses = list()
        for out_stage in out_stages:
            out_stage = out_stage.reshape((batch_size, 1, kp_num, H, W)).squeeze(1)
            losses.append(self.criterion(out_stage, target, target_weight))
        loss = paddle.mean(paddle.stack(losses, axis=0))
        keypoint_losses = {
            'loss': loss
        }
        return keypoint_losses
