# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

from ppdet.smoke_kp.skp_models.layers.transformer_utils import *
from ppdet.smoke_kp.skp_models.model_utils import load_pretrained_model

from ppdet.smoke_kp.skp_models.stack_former.stack_former_config import (
    StackFormer_B0, StackFormer_B1
)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        x_shape = paddle.shape(x)
        B, N = x_shape[0], x_shape[1]
        C = self.dim

        q = self.q(x).reshape([B, N, self.num_heads,
                               C // self.num_heads]).transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            kv = self.kv(x).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn_scores = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_scores


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 need_out_attn=False):
        super().__init__()
        self.need_out_attn = need_out_attn
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        h = x
        x, attn_scores = self.attn(self.norm1(x), H, W)
        x = h + self.drop_path(x)

        h = x
        x = self.mlp(self.norm2(x), H, W)
        x = h + self.drop_path(x)

        if self.need_out_attn:
            return x, attn_scores
        else:
            return x


class OverlapPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x)
        x_shape = paddle.shape(x)
        H, W = x_shape[2], x_shape[3]
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm(x)

        return x, H, W


class MixUpVisionTransformer(nn.Layer):
    def __init__(self,
                 num_classes=1000,
                 embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6],
                 sr_ratios=[8, 4, 2],
                 pretrained=None,
                 need_out_attn=False,
                 stack_former_config=StackFormer_B0):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.feat_channels = embed_dims[:]
        self.need_out_attn = need_out_attn
        self.align_corners = False
        self.stack_former_config = stack_former_config

        # transformer encoder
        dpr = [
            x.numpy() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        self.block_up_1 = nn.LayerList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
                need_out_attn=need_out_attn) for i in range(depths[0])
        ])
        self.norm_up_1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block_up_2 = nn.LayerList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
                need_out_attn=need_out_attn) for i in range(depths[1])
        ])
        self.norm_up_2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block_up_3 = nn.LayerList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
                need_out_attn=need_out_attn) for i in range(depths[2])
        ])
        self.norm_up_3 = norm_layer(embed_dims[2])

        self.linear_map1 = nn.Conv2D(
            self.stack_former_config['down_former_chans'][3] +
            self.stack_former_config['down_former_chans'][2],
            embed_dims[0],
            kernel_size=1
        )
        self.linear_map2 = nn.Conv2D(
            self.stack_former_config['down_former_chans'][1] + embed_dims[0],
            embed_dims[1],
            kernel_size=1
        )
        self.linear_map3 = nn.Conv2D(
            self.stack_former_config['down_former_chans'][0] + embed_dims[1],
            embed_dims[2],
            kernel_size=1
        )

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_pretrained_model(self, self.pretrained)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in paddle.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, feats):
        attn_weights = []
        outs = []
        s4, s8, s16, s32 = feats

        s4_shape = paddle.shape(s4)
        s8_shape = paddle.shape(s8)
        s16_shape = paddle.shape(s16)
        s32_shape = paddle.shape(s32)

        B = s32_shape[0]
        x = F.interpolate(
            s32,
            size=s16_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 1
        x = paddle.concat([x, s16], axis=1)
        x = self.linear_map1(x) # in_chans: s32 + s16
        x = x.reshape([0, 0, s16_shape[2] * s16_shape[3]]).transpose([0, 2, 1])
        H, W = s16_shape[2:]
        for i, blk in enumerate(self.block_up_1):
            if self.need_out_attn:
                x, attn_weight = blk(x, H, W)
                attn_weights.append(attn_weight)
            else:
                x = blk(x, H, W)
        x = self.norm_up_1(x)
        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        outs.append(x) # s16

        x = F.interpolate(
            x,
            size=s8_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 2
        x = paddle.concat([x, s8], axis=1)
        x = self.linear_map2(x) # in_chans: s8 + embed_dim[0]
        x = x.reshape([0, 0, s8_shape[2] * s8_shape[3]]).transpose([0, 2, 1])
        H, W = s8_shape[2:]
        for i, blk in enumerate(self.block_up_2):
            if self.need_out_attn:
                x, attn_weight = blk(x, H, W)
                attn_weights.append(attn_weight)
            else:
                x = blk(x, H, W)
        x = self.norm_up_2(x)
        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        outs.append(x) # s8

        x = F.interpolate(
            x,
            size=s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 3
        x = paddle.concat([x, s4], axis=1)
        x = self.linear_map3(x) # in_chans: s4 + embed_dim[1]
        x = x.reshape([0, 0, s4_shape[2] * s4_shape[3]]).transpose([0, 2, 1])
        H, W = s4_shape[2:]
        for i, blk in enumerate(self.block_up_3):
            if self.need_out_attn:
                x, attn_weight = blk(x, H, W)
                attn_weights.append(attn_weight)
            else:
                x = blk(x, H, W)
        x = self.norm_up_3(x)
        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        outs.append(x) # s4

        if self.need_out_attn:
            return outs, attn_weights
        return outs

    def forward(self, feats):
        return self.forward_features(feats)


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dim = dim
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        x_shape = paddle.shape(x)
        B, N = x_shape[0], x_shape[1]
        x = x.transpose([0, 2, 1]).reshape([B, self.dim, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose([0, 2, 1])

        return x

def MixUpVisionTransformer_B0(**kwargs):
    return MixUpVisionTransformer(
        embed_dims=StackFormer_B0['up_former_chans'],
        num_heads=StackFormer_B0['up_num_heads'],
        mlp_ratios=StackFormer_B0['up_mlp_ratios'],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2],
        sr_ratios=[8, 4, 2],
        drop_rate=0.0,
        drop_path_rate=0.1,
        stack_former_config=StackFormer_B0,
        **kwargs)


def MixUpVisionTransformer_B1(**kwargs):
    return MixUpVisionTransformer(
        embed_dims=StackFormer_B1['up_former_chans'],
        num_heads=StackFormer_B1['up_num_heads'],
        mlp_ratios=StackFormer_B1['up_mlp_ratios'],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2],
        sr_ratios=[8, 4, 2],
        drop_rate=0.0,
        drop_path_rate=0.1,
        stack_former_config=StackFormer_B1,
        **kwargs)


if __name__ == '__main__':
    mix_up = MixUpVisionTransformer_B1()
    model = paddle.Model(mix_up)
    model.summary(input_size=[
        (1, 32,  128, 128),
        (1, 64,  64,  64),
        (1, 160, 32,  32),
        (1, 256, 16,  16)
    ])
