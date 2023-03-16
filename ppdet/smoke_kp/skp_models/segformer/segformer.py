"""
@File  : segformer.py
@Author: tao.jing
@Date  : 2022/5/12
@Desc  :
"""
# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from ppdet.smoke_kp.skp_models.model_utils import load_entire_model
from ppdet.smoke_kp.skp_models import layers
from ppdet.smoke_kp.skp_models.segformer.mix_transformer import MixVisionTransformer_B0, MixVisionTransformer_B2

from ppdet.core.workspace import register, create
from ppdet.modeling.architectures.meta_arch import BaseArch

__all__ = [
    'SegFormer'
]


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
class SegFormer(BaseArch):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 num_classes,
                 embedding_dim,
                 model_level=1,
                 loss='KeyPointMSELoss',
                 post_process='HRNetPostProcess',
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.loss = loss
        self.post_process = post_process

        self.pretrained = pretrained
        self.align_corners = align_corners
        if model_level == 0:
            self.backbone = MixVisionTransformer_B0()
        else:
            self.backbone = MixVisionTransformer_B2()
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        post_process = create(cfg['post_process'])

        return {
            "post_process": post_process,
        }

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def _forward(self):
        x = self.inputs['image']

        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        outputs = logit
        '''
        outputs = F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        '''

        if self.training:
            return self.loss(outputs, self.inputs)
        else:
            imshape = (self.inputs['im_shape'].numpy()
                       )[:, ::-1] if 'im_shape' in self.inputs else None
            center = self.inputs['center'].numpy(
            ) if 'center' in self.inputs else np.round(imshape / 2.)
            scale = self.inputs['scale'].numpy(
            ) if 'scale' in self.inputs else imshape / 200.
            outputs = self.post_process(outputs, center, scale)
            return outputs


if __name__ == '__main__':
    network = SegFormer(num_classes=2, embedding_dim=768)
    paddle.summary(network, input_size=(1, 3, 512, 512))
