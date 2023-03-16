"""
@File  : stack_former_head.py
@Author: tao.jing
@Date  : 2022/5/21
@Desc  :
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@register
class StackFormerHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 embed_dim,
                 out_chans=(256, 128, 64)):
        super(StackFormerHead, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.out_chans = out_chans

        self.linear_s8 = MLP(input_dim=self.out_chans[1], embed_dim=embed_dim)
        self.linear_s4 = MLP(input_dim=self.out_chans[2], embed_dim=embed_dim)

        self.dropout = nn.Dropout2D(0.1)

        self.linear_pred_c2 = nn.Conv2D(
            embed_dim, self.num_classes, kernel_size=1)
        self.linear_pred_c1 = nn.Conv2D(
            embed_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        s16, s8, s4 = x
        s4_shape = paddle.shape(s4)
        s8_shape = paddle.shape(s8)

        s8 = self.linear_s8(s8).transpose([0, 2, 1]).reshape(
            [0, 0, s8_shape[2], s8_shape[3]])

        s4 = self.linear_s4(s4).transpose([0, 2, 1]).reshape(
            [0, 0, s4_shape[2], s4_shape[3]])

        logit_s4 = self.dropout(s4)
        logit_s8 = self.dropout(s8)
        logit_s4 = self.linear_pred_c1(logit_s4)
        logit_s8 = self.linear_pred_c2(logit_s8)

        '''
        out_s4 = F.interpolate(
            logit_s8,
            size=s4_shape[2:],
            mode='bilinear',
            align_corners=False)

        out_s2 = F.interpolate(
            logit_s4,
            size=s4_shape[2:] * 2,
            mode='bilinear',
            align_corners=False)

        outputs = (out_s2, out_s4)
        return outputs
        '''
        return [logit_s4, logit_s8]