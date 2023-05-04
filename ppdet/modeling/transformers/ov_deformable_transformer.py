# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, get_valid_ratio
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

from ..embedder.clip_utils import build_text_embedding_coco

import pickle

__all__ = ['OVDeformableTransformer',
           # 'OVDeformableTransformerEncoder',
           # 'OVDeformableTransformerDecoder',
           ]

class MLP(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# 对齐
class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points, bias_attr=True)
        # TODO check embed_dim 256*3 or 256
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)

        self.output_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        try:
            # use cuda op
            from deformable_detr_ops import ms_deformable_attn
        except:
            # use paddle func
            from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = ms_deformable_attn

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
                or [bs, query_length, n_levels, 4), add additional (w, h) to form reference boxes
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                    format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


# 对齐
class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes,
                level_start_index,
                src_mask=None,
                pos_embed=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos_embed), reference_points, src,
            spatial_shapes, level_start_index, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src


# 对齐
# @register
# class OVDeformableTransformerEncoder(nn.Layer):
#     def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, activation,
#                 num_feature_levels, num_encoder_points, num_layers, weight_attr=None, bias_attr=None):
#         super(OVDeformableTransformerEncoder, self).__init__()
#         # encoder_layer = DeformableTransformerEncoderLayer(
#         #     hidden_dim, nhead, dim_feedforward, dropout, activation,
#         #     num_feature_levels, num_encoder_points, weight_attr, bias_attr)
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         # self.hidden_dim = hidden_dim
#         # self.nhead = nhead
class OVDeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(OVDeformableTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, offset=0.5):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.arange(end=H) + offset, paddle.arange(end=W) + offset)
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] *
                                                    H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] *
                                                    W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                level_start_index,
                src_mask=None,
                pos_embed=None,
                valid_ratios=None):
        output = src
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [src.shape[0], spatial_shapes.shape[0], 2])
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for layer in self.layers:
            output = layer(output, reference_points, spatial_shapes,
                           level_start_index, src_mask, pos_embed)

        return output


# 对齐
class DeformableTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                query_pos_embed=None,
                tgt_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


# 对齐
# @register
# class OVDeformableTransformerDecoder(nn.Layer):
#     def __init__(self,
#                  nhead,
#                  dim_feedforward,
#                  dropout,
#                  activation,
#                  num_feature_levels,
#                  num_decoder_points,
#                  num_layers,
#                  cls_out_channels=1,
#                  hidden_dim=256,
#                  num_mlp_layers=3,
#                  return_intermediate=False,
#                  with_box_refine=False,
#                  weight_attr=None,
#                  bias_attr=None):
#         super(OVDeformableTransformerDecoder, self).__init__()
#         decoder_layer = DeformableTransformerDecoderLayer(
#             hidden_dim, nhead, dim_feedforward, dropout, activation,
#             num_feature_levels, num_decoder_points, weight_attr, bias_attr)
class OVDeformableTransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, cls_out_channels,
                 return_intermediate=False,
                 ):
        super(OVDeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.cls_out_channel = cls_out_channels
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None
        self.score_head = None

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_valid_ratios,
                memory_mask=None,
                query_pos_embed=None,
                tgt_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * paddle.concat([memory_valid_ratios, memory_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points.unsqueeze(2) * src_valid_ratios.unsqueeze(1)
            output = layer(output, reference_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           memory_mask, query_pos_embed, tgt_mask=tgt_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_head is not None:
                tmp = self.bbox_head[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = F.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = F.sigmoid(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(intermediate_reference_points)

        return output.unsqueeze(0)


@register
class OVDeformableTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 two_stage=True,
                 two_stage_num_proposals=300,
                 with_box_refine=False,
                 cls_out_channels=2,
                 max_len=15,
                 clip_feat_path='/home/a401-2/PycharmProjects/PaddleDetection/ppdet/modeling/transformers/clip_feat_coco_pickle_label.pkl',
                 prob=0.5,
                 zeroshot_w=build_text_embedding_coco(),
                 # head = 'OVDeformableDETRHead',
                 weight_attr=None,
                 bias_attr=None):
        super(OVDeformableTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_encoder_points, weight_attr, bias_attr)
        self.encoder = OVDeformableTransformerEncoder(encoder_layer,
                                                      num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_decoder_points, weight_attr, bias_attr)
        self.decoder = OVDeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, cls_out_channels,
            return_intermediate=return_intermediate_dec)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim, bias_attr=True)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            self.pos_trans = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias_attr=True)
            self.pos_trans_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
            # self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            self.reference_points = nn.Linear(
                hidden_dim,
                2,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult))
            normal_(self.query_embed.weight)

        # self.input_proj = nn.LayerList()
        # for in_channels in backbone_num_channels:
        #     self.input_proj.append(
        #         nn.Sequential(
        #             nn.Conv2D(
        #                 in_channels,
        #                 hidden_dim,
        #                 kernel_size=1,
        #                 weight_attr=weight_attr,
        #                 bias_attr=bias_attr),
        #             nn.GroupNorm(32, hidden_dim)))
        # in_channels = backbone_num_channels[-1]
        # for _ in range(num_feature_levels - len(backbone_num_channels)):
        #     self.input_proj.append(
        #         nn.Sequential(
        #             nn.Conv2D(
        #                 in_channels,
        #                 hidden_dim,
        #                 kernel_size=3,
        #                 stride=2,
        #                 padding=1,
        #                 weight_attr=weight_attr,
        #                 bias_attr=bias_attr),
        #             nn.GroupNorm(32, hidden_dim)))
        #     in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=-0.5)

        # self.zeroshot_w = zeroshot_w.t()

        # self.patch2query = nn.Linear(512, 256)
        # self.patch2query_img = nn.Linear(512, 256)
        # # mark 源码此处for layer in [self.patch2query]:
        # xavier_uniform_(self.patch2query.weight)
        # constant_(self.patch2query.bias, 0)

        # # self.head = head
        # num_pred = self.decoder.num_layers
        #
        # self.all_ids = paddle.to_tensor(list(range(self.zeroshot_w.shape[-1])))
        # self.max_len = max_len
        # self.max_pad_len = max_len - 3
        #
        # with open(clip_feat_path, 'rb') as f:
        #     self.clip_feat = pickle.load(f)
        # self.prob = prob
        #
        # self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        # normal_(self.tgt_embed.weight)
        # normal_(self.query_pos_embed.weight)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight)
            constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_num_channels': [i.channels for i in input_shape], }

    def get_proposal_pos_embed(self, proposals):
        # print(proposals.shape)
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        # mark diff 不需要梯度回传待查
        # dim_t = paddle.arange(num_pos_feats, dtype='float32').requires_grad_(False)
        # with paddle.no_grad():
        #     dim_t = paddle.arange(num_pos_feats)
        dim_t = paddle.arange(num_pos_feats)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats).astype('float32')
        # mark diff < 1e-8
        # print(dim_t)
        # N, L, 4
        # proposals = proposals.sigmoid() * scale
        proposals = F.sigmoid(proposals) * scale
        # print(proposals.shape)
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # print(pos.shape)
        # N, L, 4, 64, 2
        pos = paddle.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), axis=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        memory_padding_mask = memory_padding_mask.astype('bool')
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].reshape((N_, H_, W_, 1))
            valid_H = paddle.sum(mask_flatten_[:, :, 0, 0], 1)
            valid_W = paddle.sum(mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = paddle.meshgrid(paddle.linspace(0, H_ - 1, H_, 'float32'),
                                             paddle.linspace(0, W_ - 1, W_, 'float32'))
            grid = paddle.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = paddle.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape((N_, 1, 1, 2))
            grid = (paddle.expand(grid.unsqueeze(0), [N_, -1, -1, -1]) + 0.5) / scale

            wh = paddle.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = paddle.concat((grid, wh), -1).reshape((N_, -1, 4))
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = paddle.concat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = paddle.log(output_proposals / (1 - output_proposals))
        output_proposals = masked_fill(output_proposals, ~memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = masked_fill(output_proposals, ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = masked_fill(output_memory, ~memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = masked_fill(output_memory, ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, src_feats,
                src_mask=None,
                inputs=None,
                query_embed=None,
                text_query=None,
                *args, **kwargs):
        # srcs = []
        # for i in range(len(src_feats)):
        #     srcs.append(self.input_proj[i](src_feats[i]))
        # if self.num_feature_levels > len(srcs):
        #     len_srcs = len(srcs)
        #     for i in range(len_srcs, self.num_feature_levels):
        #         if i == len_srcs:
        #             srcs.append(self.input_proj[i](src_feats[-1]))
        #         else:
        #             srcs.append(self.input_proj[i](srcs[-1]))
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(src_feats):
            bs, _, h, w = paddle.shape(src)
            spatial_shapes.append(paddle.concat([h, w]))
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            pos_embed = self.position_embedding(mask).flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[level]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        # print('src_flatten', src_flatten)
        # TODO 待考究
        # mask_flatten = None if src_mask is None else paddle.concat(mask_flatten,
        #                                                            1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [l, 2]
        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype('int64'))
        # [l], 每一个level的起始index
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, l, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)
        # print('memory', memory)

        # prepare input for decoder
        bs, _, c = memory.shape
        # done
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.score_head[6](output_memory)
            # print('output_proposals',output_proposals)
            # print('self.decoder.bbox_head(output_memory)', self.decoder.bbox_head(output_memory))
            enc_outputs_coord_unact = self.decoder.bbox_head[6](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = paddle.topk(enc_outputs_class[..., 0], topk, axis=1)[1]

            topk_coords_unact = paddle.take_along_axis(enc_outputs_coord_unact,
                                                       paddle.tile(topk_proposals.unsqueeze(-1), (1, 1, 4)), 1)
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = F.sigmoid(topk_coords_unact)
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = paddle.split(pos_trans_out, int(pos_trans_out.shape[-1] / c), axis=2)

            num_queries = query_embed.shape[1]
            num_patch = len(text_query)
            query_embed = paddle.tile(query_embed, (1, num_patch, 1))

            tgt = paddle.tile(tgt, (1, num_patch, 1))
            text_query = text_query.repeat_interleave(num_queries, 0)
            text_query = paddle.expand(text_query.unsqueeze(0), (bs, -1, -1))
            tgt = tgt + text_query
            reference_points = paddle.tile(reference_points, (1, num_patch, 1))
            init_reference_out = paddle.tile(init_reference_out, (1, num_patch, 1))
        else:
            query_embed, tgt = paddle.split(query_embed, int(query_embed.shape[-1] / c), axis=1)
            num_queries = len(query_embed)
            num_patch = len(text_query)
            query_embed = paddle.tile(query_embed, (num_patch, 1))
            query_embed = paddle.expand(query_embed.unsqueeze(0), (bs, -1, -1))
            tgt = paddle.tile(tgt, (num_patch, 1))
            tgt = paddle.expand(tgt.unsqueeze(0), (bs, -1, -1))
            text_query = paddle.repeat_interleave(text_query, num_queries, 0)
            text_query = paddle.expand(text_query.unsqueeze(0), (bs, -1, -1))
            tgt = tgt + text_query
            reference_points = F.sigmoid(self.reference_points(query_embed))
            init_reference_out = reference_points

        decoder_mask = (
                paddle.ones([num_queries * num_patch, num_queries * num_patch])
                * float("-inf")
        )
        for i in range(num_patch):
            decoder_mask[
            i * num_queries: (i + 1) * num_queries,
            i * num_queries: (i + 1) * num_queries,
            ] = 0

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes,
                                            level_start_index, valid_ratios, mask_flatten, query_embed, decoder_mask)

        memory_features = []
        spatial_index = 0
        for lvl in range(len(spatial_shapes)):
            h, w = spatial_shapes[lvl]
            memory_lvl = (paddle.transpose(
                paddle.reshape(
                    memory[:, spatial_index: spatial_index + h * w, :],
                    (bs, h, w, c)), perm=(0, 3, 1, 2)))
            memory_features.append(memory_lvl)
            spatial_index += h * w

        # inter_references_out = inter_references
        references = [reference_points, *inter_references]

        head_inputs_dict = dict(
            feats=hs,
            reference=references,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_unact=enc_outputs_coord_unact,
        )
        if self.two_stage:
            return head_inputs_dict
            # return (hs,
            #         init_reference_out,
            #         inter_references_out,
            #         enc_outputs_class,
            #         enc_outputs_coord_unact,
            #         ), (select_id, clip_query_ori), memory_features
        return (hs, init_reference_out, inter_references_out, None, None), memory_features
        # return (hs, memory, reference_points)


def masked_fill(tensor, mask, value):
    cover = paddle.full_like(tensor, value)
    out = paddle.where(mask, cover, tensor)
    return out


def inverse_sigmoid(x, eps=1e-5):
    x = paddle.clip(x, min=0, max=1)
    x1 = paddle.clip(x, min=eps)
    x2 = paddle.clip(1 - x, min=eps)
    return paddle.log(x1 / x2)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_ov_deformable_transfomer():
    return OVDeformableTransformer()


# 单元测试
if __name__ == '__main__':
    set_seed(7)

    #字典转结构体
    # class DictToStruct:
    #     def __init__(self, **entries):
    #         self.__dict__.update(entries)

    #COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # params = {"hidden_dim": 256,
    #           "dec_n_points": 4,
    #           "dim_feedforward": 1024,
    #           "dropout": 0.1,
    #           "enc_n_points": 4,
    #           "nheads": 8,
    #           "enc_layers": 6,
    #           "dec_layers": 6,
    #           "num_feature_levels": 4,
    #           "two_stage": False,
    #           "num_queries": 300
    #           }
    # args = DictToStruct(**params)
    trans = build_ov_deforamble_transformer()


