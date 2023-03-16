"""
@File  : joints_mse_loss.py
@Author: tao.jing
@Date  : 2022/12/16
@Desc  :
"""
import numpy as np
import paddle
from paddle import nn
from ppdet.core.workspace import register, serializable


__all__ = [
    'SKPJointsMSELoss'
]


@register
@serializable
class SKPJointsMSELoss(nn.Layer):
    def __init__(self, num_joints, hmsize, use_target_weight=False):
        super(SKPJointsMSELoss, self).__init__()
        self.num_joints = num_joints
        self.hmsize = np.array(hmsize)
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, records, target_weight=None):
        output = output[0]
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        loss = 0

        for idx, hmsize in enumerate(self.hmsize):
            heatmaps = records['heatmap_gt{}x'.format(idx + 1)]
            heatmaps_gt = heatmaps.reshape((batch_size, num_joints, -1))

            for idx in range(num_joints):
                heatmap_pred = heatmaps_pred[:, idx, :].squeeze()
                heatmap_gt = heatmaps_gt[:, idx, :].squeeze().astype(paddle.float32)
                if self.use_target_weight and target_weight:
                    loss += 0.5 * self.criterion(
                        heatmap_pred.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx])
                    )
                else:
                    loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        keypoint_losses = {
            'loss': loss
        }
        return keypoint_losses