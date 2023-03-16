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
from ppdet.smoke_kp.skp_models.stack_former.mix_transformer import (
    MixVisionTransformer_B0, MixVisionTransformer_B1
)
from ppdet.smoke_kp.skp_models.stack_former.mix_up_transformer import (
    MixUpVisionTransformer_B0, MixUpVisionTransformer_B1
)
from ppdet.smoke_kp.skp_models.stack_former.stack_former_head import StackFormerHead
from ppdet.modeling import HrHRNetHead

from ppdet.core.workspace import register, create
from ppdet.modeling.architectures.meta_arch import BaseArch

__all__ = [
    'StackFormer'
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
class StackFormer(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 model_level=0,
                 loss='MultiHmMSELoss',
                 post_process='HRNetPostProcess',
                 former_head='StackFormerHead',
                 pretrained=None,
                 preview_model=False):
        super(StackFormer, self).__init__()

        self.model_level = model_level

        self.loss = loss
        self.post_process = post_process

        self.pretrained = pretrained
        self.preview_model = preview_model

        if self.model_level == 0:
            self.down_former = MixVisionTransformer_B0()
            self.up_former = MixUpVisionTransformer_B0()
        elif self.model_level == 1:
            self.down_former = MixVisionTransformer_B1()
            self.up_former = MixUpVisionTransformer_B1()

        self.out_chans = self.up_former.feat_channels

        if not self.preview_model:
            if isinstance(former_head, StackFormerHead):
                self.use_hrhr_head = False
                self.former_head = type(former_head)(
                    former_head.num_classes,
                    former_head.embed_dim,
                    self.out_chans
                )
            elif isinstance(former_head, HrHRNetHead):
                self.use_hrhr_head = True
                self.former_head = former_head
            else:
                raise ValueError(f'Invalid former_head type.')
        else:
            self.former_head = StackFormerHead(
                num_classes = 2,
                embed_dim = 256,
                out_chans = self.out_chans
            )
        self.interpolate = nn.Upsample(None, 2, mode='bilinear')
        self.pool = nn.MaxPool2D(5, 1, 2)
        self.max_num_people = 1

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        post_process = create(cfg['post_process'])
        former_head = create(cfg['former_head'], **kwargs)

        return {
            "post_process": post_process,
            "former_head": former_head,
        }

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def get_topk(self, outputs):
        # resize to image size
        outputs = [self.interpolate(x) for x in outputs]
        if len(outputs) == 3:
            tagmap = paddle.concat(
                (outputs[1].unsqueeze(4), outputs[2].unsqueeze(4)), axis=4)
        else:
            tagmap = outputs[1].unsqueeze(4)

        heatmap = outputs[0]
        N, J = 1, self.former_head.num_joints
        heatmap_maxpool = self.pool(heatmap)
        # topk
        maxmap = heatmap * (heatmap == heatmap_maxpool)
        maxmap = maxmap.reshape([N, J, -1])
        heat_k, inds_k = maxmap.topk(self.max_num_people, axis=2)

        outputs = [heatmap, tagmap, heat_k, inds_k]
        return outputs

    def _forward(self):
        x = self.inputs['image']

        feats = self.down_former(x)
        outs = self.up_former(feats)

        if not self.use_hrhr_head:
            outs = self.former_head(outs)

            if self.training:
                return self.loss(outs, self.inputs)
            else:
                imshape = (self.inputs['im_shape'].numpy()
                           )[:, ::-1] if 'im_shape' in self.inputs else None
                center = self.inpuets['center'].numpy(
                ) if 'center' in self.inputs else np.round(imshape / 2.)
                scale = self.inputs['scale'].numpy(
                ) if 'scale' in self.inputs else imshape / 200.
                outputs = self.post_process(outs[0], center, scale)
                return outputs
        else:
            if self.training:
                outs = [outs[-1], ]
                return self.former_head(outs, self.inputs)
            else:
                outs = [outs[-1], ]
                outputs = self.former_head(outs)
                outputs = self.get_topk(outputs)

                res_lst = []
                h = self.inputs['im_shape'][0, 0].numpy().item()
                w = self.inputs['im_shape'][0, 1].numpy().item()
                kpts, scores = self.post_process(*outputs, h, w)
                res_lst.append([kpts, scores])
                return res_lst


if __name__ == '__main__':
    network = StackFormer(preview_model=True)
    paddle.summary(network, input_size=(1, 3, 512, 512))
