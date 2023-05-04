# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# This code is based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant, Assign

from ppdet.modeling.layers import MultiHeadAttention
from ppdet.modeling.initializer import zeros_, normal_
from ppdet.core.workspace import register

from .models import ModifiedResNet, ViT
from .layers import Transformer, LayerNorm, QuickGELU, AttentionPool2D
from .tokenizer import tokenize


@register
class CLIP(nn.Layer):
    def __init__(self,
                 embed_dim=512,
                 # vision
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=32,
                 # text
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = ViT(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))
            )
        )
        self.add_parameter("positional_embedding", positional_embedding)

        self.ln_final = nn.LayerNorm(transformer_width)

        text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))
            )
        )
        self.add_parameter("text_projection", text_projection)
        # self.text_projection = nn.Linear(
        #     transformer_width, embed_dim, bias_attr=False)

        logit_scale = self.create_parameter(
            shape=(1,),
            default_initializer=Assign(paddle.ones([1]))
        )
        self.add_parameter("logit_scale", logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.embed_dim ** -0.5
                # normal_ = Normal(std=std)
                normal_(self.visual.attnpool.attn.q_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.k_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.v_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.out_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        Constant(value=0.0)(param)

        proj_std = (self.transformer.width ** -0.5) * \
            ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for resblock in self.transformer.resblocks:
            # normal_ = Normal(std=attn_std)
            normal_(resblock.attn.q_proj.weight, std=attn_std)
            normal_(resblock.attn.k_proj.weight, std=attn_std)
            normal_(resblock.attn.v_proj.weight, std=attn_std)
            normal_(resblock.attn.out_proj.weight, std=proj_std)
            normal_(resblock.mlp.c_fc.weight, std=fc_std)
            normal_(resblock.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            # normal_(
            #     self.text_projection.weight,
            #     std=self.transformer.width ** -0.5)
            Normal(std=self.transformer.width ** -0.5)(self.text_projection)

    def build_attention_mask(self):
        mask = paddle.full(
            (self.context_length, self.context_length), float("-inf")
        )
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        select = []
        index = zip(
            paddle.arange(x.shape[0]).numpy(),
            text.argmax(axis=-1).numpy()
        )
        for i, j in index:
            select.append(x[int(i), int(j)])

        x = paddle.stack(select) @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

COCO_CATEGORIES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "computer mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush",
}

def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]

# class LoadClipQuery(nn.Layer):
#     __inject__ = ['text_encoder']
#
#     def __int__(self,
#                 clip_feat_path,
#                 text_encoder):
#         super(LoadClipQuery, self).__int__()
#         self.clip_feat_path = clip_feat_path
#         self.text_encoder = text_encoder

# def build_text_embedding_coco():
#     categories = COCO_CATEGORIES
#     run_on_gpu = True
#
#     clip_model = CLIP()
#     text_model = clip_model.text()
#
#     for name, param in text_model.named_parameters():
#         param.requires_grad = False
#     templates = multiple_templates
#     with paddle.no_grad():
#         zeroshot_weights = []
#         for _, category in categories.items():
#             texts = [
#                 template.format(processed_name(category, rm_dot=True), article=article(category))
#                 for template in templates
#             ]
#             texts = [
#                 "This is " + text if text.startswith("a") or text.startswith("the") else text
#                 for text in texts
#             ]
#             texts = tokenize(texts)  # tokenize
#             if run_on_gpu:
#                 texts = texts.cuda()
#             text_embeddings = text_model(texts)
#             text_embeddings /= paddle.linalg.norm(text_embeddings, axis=-1, keepdim=True)
#             text_embedding = paddle.mean(text_embeddings, axis=0)
#             text_embedding /= paddle.linalg.norm(text_embedding)
#             zeroshot_weights.append(text_embedding)
#         zeroshot_weights = paddle.stack(zeroshot_weights, axis=1)
#         if run_on_gpu:
#             zeroshot_weights = zeroshot_weights.cuda()
#     zeroshot_weights = zeroshot_weights.t().numpy()
#     all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
#                36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73,
#                74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
#     all_ids = [i - 1 for i in all_ids]
#     return paddle.to_tensor(zeroshot_weights[all_ids])


class embedder(nn.Layer):
    def __init__(self):
        super(embedder, self).__init__()
