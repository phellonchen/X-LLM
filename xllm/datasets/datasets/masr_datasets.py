"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import numpy as np
from collections import OrderedDict

from xllm.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class MSpeech2TextDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
            vis_root (string): Root directory of images (e.g. coco/images/)
            ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        split = 'val'
        speech_feats_path = ann["speech_feats_path"]
        # if 'aishell2' in speech_feats_path:
        #     speech_feats_path = speech_feats_path.split('/')[-1]
        #     speech_feats_path = os.path.join('/raid/cfl/cn_pretraining_multi_dialog/data/aishell2/aishell2_shards', speech_feats_path)
        # else:
        #     if 'train' in speech_feats_path:
        #         speech_feats_path = speech_feats_path.split('/')[-1]
        #         speech_feats_path = os.path.join('/raid/cfl/cn_pretraining_multi_dialog/data/visdial_cn/vsdial_cn_cif_feats/shards/train', speech_feats_path)
        #     else:
        #         speech_feats_path = speech_feats_path.split('/')[-1]
        #         speech_feats_path = os.path.join('/raid/cfl/cn_pretraining_multi_dialog/data/visdial_cn/vsdial_cn_cif_feats/shards/val', speech_feats_path)
        #         split = 'val'
 
        speech_feats = np.load(speech_feats_path)   # L x D, ndarray type with float type
        # speech_feats = np.expand_dims(speech_feats, axis=0)

        target_text = self.text_processor(ann["target_text"])
        source_text, target_text = self.corrupt_prefix_zh(target_text, split)

        # give it an random default image
        image_path = os.path.join("/raid/cfl/pretraining/en/data/images/VisualDialog_test2018", ann["image"].split("/")[-1])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # truncate text and pad speech feats
        max_length = 20
        if len(list(target_text)) > max_length:
            target_text = "".join(list(target_text)[:max_length])
        if speech_feats.shape[0] > max_length:
            speech_feats = speech_feats[:max_length, :]
        if speech_feats.shape[0] < max_length:
            speech_feat_len = speech_feats.shape[0]
            pad_len = max_length - speech_feat_len
            speech_feats = np.pad(
                speech_feats,
                ((0,pad_len), (0,0)),
                "constant",
                constant_values=(0,0)
            )   # print(speech_feats, speech_feats.shape)

        return {
            "image": image,
            "text_input": source_text,
            "speech_input": speech_feats,
            "text_output": target_text,
            "task_id": 0,
        }
    
    def corrupt_prefix(self, input_text):
        """T5-style Prefix Language Modeling
        Args:
            text
        Returns:
            source_text (prefix)
            target_text
        Ex) (in vocab ids)
        input
            In this tutorial, we’ll explore how to preprocess your data using Transformers.
            The main tool for this is what we call a tokenizer.
        source text
            this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool
        target_text
            for this is what we call a tokenizer.
        """

        tokens = input_text.split()

        n_tokens = len(tokens)
        split = random.randint(1, n_tokens - 1)
        source_text = " ".join(tokens[:split])
        target_text = " ".join(tokens[split:])

        return source_text, target_text
    
    def corrupt_prefix_zh(self, input_text, split='train'):
        """T5-style Prefix Language Modeling
        Args:
            text
        Returns:
            source_text (prefix)
            target_text
        Ex) (in vocab ids)
        input
            In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.
        source text
            this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool
        target_text
            for this is what we call a tokenizer.
        """

        tokens = list(input_text)
        if split == 'train':
            source_text, target_text = "请忠实地识别该语音[gMASK]", "".join(tokens)
        else:
            source_text, target_text = "根据图片请忠实地识别该语音", "".join(tokens)
        return source_text, target_text

# 请重复前面语音中的文本