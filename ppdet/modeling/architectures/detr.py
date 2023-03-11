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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

from ..embedder.clip_utils import build_text_embedding_coco

__all__ = ['DETR', 'OVDETR']


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 detr_head,
                 post_process='DETRBBoxPostProcess',
                 exclude_post_process=False):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # transformer
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        pad_mask = self.inputs['pad_mask'] if self.training else None
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            return self.detr_head(out_transformer, body_feats, self.inputs)
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bboxes, logits, masks = preds
                return bboxes, logits
            else:
                bbox, bbox_num = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output

@register
class OVDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['exclude_post_process']

    def __init__(self,
                 backbone,
                 neck,
                 encoder,
                 decoder,
                 detr_head,
                 # zeroshot_w,
                 two_stage,
                 with_box_refine,
                 num_feature_levels,
                 post_process='DETRBBoxPostProcess',
                 exclude_post_process=False):
        super(OVDETR, self).__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
        self.encoder = encoder
        self.decoder = decoder
        self.detr_head = detr_head
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process

        # pre_encoder
        self.position_embedding = position_embedding
        # pre_encoder
        # self.position_embedding = PositionEmbedding(
        #     hidden_dim // 2,
        #     normalize=True if position_embed_type == 'sine' else False,
        #     embed_type=position_embed_type,
        #     offset=-0.5)
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

        # self.zeroshot_w = zeroshot_w.t()
        self.patch2query = nn.Linear(512, 256)
        self.patch2query_img = nn.Linear(512, 256)
        # mark 源码此处for layer in [self.patch2query]:
        xavier_uniform_(self.patch2query.weight)
        constant_(self.patch2query.bias, 0)

        num_pred = self.decoder.num_layers
        self.all_ids = paddle.to_tensor(list(range(self.zeroshot_w.shape[-1])))
        self.max_len = max_len
        self.max_pad_len = max_len - 3

        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim, bias_attr=True)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            self.pos_trans = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias_attr=True)
            self.pos_trans_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            self.reference_points = nn.Linear(
                hidden_dim,
                2,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult))
            normal_(self.query_embed.weight)
            self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        # normal_(self.tgt_embed.weight)
        # normal_(self.query_pos_embed.weight)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight)
            constant_(self.reference_points.bias)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # transformer
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # zeroshot_w = create(cfg['embedder'])

        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        body_feats = self.neck(bodey_feats)

        pad_mask = self.inputs['pad_mask'] if self.training else None
        print(self.inputs)
        # out_transformer, clip_id, memory_feature = self.transformer(body_feats, pad_mask, self.inputs)
        # Transformer
        encoder_outputs_dict = self.forward_ov_transformer(body_feats)

        # DETR Head
        if self.training:
            return self.detr_head(out_transformer, clip_id, memory_feature, body_feats, self.inputs)
        else:
            preds = self.detr_head(out_transformer, memory_feature, body_feats)
            if self.exclude_post_process:
                bboxes, logits, masks = preds
                return bboxes, logits
            else:
                bbox, bbox_num = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output

    def forward_ov_transformer(self, inputs):
        encoder_inputs_dict, decoder_inputs_dict = self.pre_encoder(inputs)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        decoder_inputs_dict_tmp, head_inputs_dict = self.pre_decoder(**decoder_inputs_dict)
        decoder_inputs_dict.update(decoder_inputs_dict_tmp)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def pre_encoder(self, srcs):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
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
        mask_flatten = None if src_mask is None else paddle.concat(mask_flatten,
                                                                   1)
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
        encoder_inputs_dict = dict(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            src_mask=mask_flatten,
            src_pos=lvl_pos_embed_flatten,
            valid_ratios=valid_ratios,
        )
        decoder_inputs_dict = dict(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self,
                        src,
                        spatial_shape,
                        level_start_index,
                        src_mask,
                        src_pos,
                        valid_ratios,
                        ):
        memory = self.encoder(
            src,spatial_shape,level_start_index,
            src_mask,src_pos,valid_ratios,
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=src_mask,
            spatial_shape=spatial_shape,
        )
        return encoder_outputs_dict

    def pre_decoder(self,
                    # decoder
                    memory,
                    mask_flatten,
                    spatial_shapes,
                    ):
        # prepare for clip_query
        if self.training:
            # uniq_labels = paddle.concat([t["gt_class"] for t in inputs])
            uniq_labels = paddle.concat(inputs["gt_class"])
            uniq_labels = paddle.unique(uniq_labels)
            uniq_labels = uniq_labels[paddle.randperm(len(uniq_labels))][: self.max_len]
            # uniq_labels = uniq_labels[paddle.to_tensor(list(range(len(uniq_labels))))][: self.max_len]
            select_id = uniq_labels.tolist()
            # mark 添加补齐id
            if len(select_id) < self.max_pad_len:
                pad_len = self.max_pad_len - len(uniq_labels)
                extra_list = [i for i in self.all_ids if i not in uniq_labels]
                extra_list = paddle.to_tensor(extra_list)
                extra_labels = extra_list[paddle.randperm(len(extra_list))][:pad_len].squeeze(1)
                # extra_labels = extra_list[paddle.to_tensor(list(range(len(extra_list))))][:pad_len].squeeze(1)
                select_id += extra_labels.tolist()
            select_id_tensor = paddle.to_tensor(select_id)
            text_query = paddle.index_select(self.zeroshot_w, select_id_tensor, axis=1).t()
            img_query = []
            for cat_id in select_id:
                index = paddle.randperm(len(self.clip_feat[cat_id]))[0:1]
                # index = paddle.to_tensor(list(range(len(self.clip_feat[cat_id]))))[0:1]
                img_query.append(paddle.to_tensor(self.clip_feat[cat_id]).index_select(index))
            img_query = paddle.concat(img_query)
            img_query = img_query / paddle.linalg.norm(img_query, axis=1, keepdim=True)

            mask = (paddle.rand([len(text_query)]) < self.prob).astype('float16').unsqueeze(1)
            # mask = (paddle.zeros([len(text_query)]) < self.prob).astype('float16').unsqueeze(1)
            clip_query_ori = (text_query * mask + img_query * (1 - mask)).detach()

            dtype = self.patch2query.weight.dtype
            text_query = self.patch2query(text_query.astype(dtype))
            img_query = self.patch2query_img(img_query.astype(dtype))
            clip_query = text_query * mask + img_query * (1 - mask)
        else:
            select_id = list(range(self.zeroshot_w.shape[-1]))
            num_patch = 15
            dtype = self.patch2query.weight.dtype
            for c in range(len(select_id) // num_patch + 1):
                clip_query = self.zeroshot_w[:, c * num_patch: (c + 1) * num_patch].t()
                clip_query = self.patch2query(clip_query.astype(dtype))

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        # prepare input for decoder
        bs, _, c = memory.shape
        # done
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.detr_head.score_head(output_memory)
            print('output_proposals', output_proposals)
            print('self.decoder.bbox_head(output_memory)', self.decoder.bbox_head(output_memory))
            enc_outputs_coord_unact = self.detr_head.bbox_head(output_memory) + output_proposals
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
            num_patch = len(clip_query)
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

        # TODO 校对输入输出
        decoder_inputs_dict = dict(
            tgt=tgt,
            query_embeds=query_embeds,
            memory=memory,
            reference_points=reference_points,
        )
        head_inputs_dict=dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_unact=enc_outputs_coord_unact,
        )
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            mask_flatten, query_embed):
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            mask_flatten,
            query_embed,
            decoer_mask,
            bbox_embed,
        )
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hs=hs, references=references
        )
        return decoder_outputs_dict


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
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].astype('bool').reshape((N_, H_, W_, 1))
            valid_H = paddle.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = paddle.sum(~mask_flatten_[:, 0, :, 0], 1)

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
        output_proposals = masked_fill(output_proposals, memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = masked_fill(output_proposals, ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = masked_fill(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = masked_fill(output_memory, ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

def masked_fill(tensor, mask, value):
    cover = paddle.full_like(tensor, value)
    out = paddle.where(mask, cover, tensor)
    return out