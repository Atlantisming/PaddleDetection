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
import os
import wget
import paddle
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .clip_models import CLIP
from .tokenizer import tokenize


def get_transforms(image_resolution):
    transforms = Compose([
        Resize(image_resolution, interpolation='bicubic'),
        CenterCrop(image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transforms


def clip_rn50():
    model = CLIP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


def clip_rn101():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 23, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


def clip_rn50x4():
    model = CLIP(
        embed_dim=640,
        image_resolution=288,
        vision_layers=(4, 6, 10, 6),
        vision_width=80,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=640,
        transformer_heads=10,
        transformer_layers=12
    )
    return model, get_transforms(288)


def clip_vit_b_32():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


model_dict = {
    'RN50': [clip_rn50, r'https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN50.pdparams', 'RN50.pdparams'],
    'RN50x4': [clip_rn50x4, r'https://bj.bcebos.com/v1/ai-studio-online/9f874e0174da48ffbd7c17e77b1fb278632620a9995e476ba873e334caec9037?responseContentDisposition=attachment%3B%20filename%3DRN50x4.pdparams', 'RN50x4.pdparams'],
    'RN101': [clip_rn101, r'https://bj.bcebos.com/v1/ai-studio-online/484592d98c584785bc8f6f9f7badbf4a9fb7a96f6102470697ed974e8eeee2a9?responseContentDisposition=attachment%3B%20filename%3DRN101.pdparams', 'RN101.pdparams'],
    'ViT_B_32': [clip_vit_b_32, r'https://bj.bcebos.com/v1/ai-studio-online/eb5e4dbf1ec142caa003a27cefd510ef46a8a6c3932a4d60bfecb3f3ab746c02?responseContentDisposition=attachment%3B%20filename%3DViT-B-32.pdparams', 'ViT-B-32.pdparams']
}


def load_model(model_name, pretrained=False):
    model_fn, url, file_name = model_dict[model_name]
    model, transforms = model_fn()

    if pretrained:
        model_path = os.path.join('pretrained_models', file_name)
        if not os.path.isfile(model_path):
            if not os.path.exists('pretrained_models'):
                os.mkdir('pretrained_models')
            wget.download(url, out=model_path)
        params = paddle.load(model_path)
        model.set_state_dict(params)
    model.eval()
    return model, transforms