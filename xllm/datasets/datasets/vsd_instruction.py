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


class VSDIntructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
            vis_root (string): Root directory of images (e.g. coco/images/)
            ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompt = [
            '请根据图像回答语音中的问题',
            '请利用图像回答语音中的问题',
            '请根据图像直接回答语音中的问题',
            '请利用图片回答语音中的问题',
        ]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        split = 'val'
        speech_feats_path = ann["speech_feats_path"]

 
        speech_feats = np.load(speech_feats_path)   # L x D, ndarray type with float type
        # speech_feats = np.expand_dims(speech_feats, axis=0)

        target_text = self.text_processor(ann["answer_text"])
        source_text = random.choice(self.prompt)

        # question_text = self.text_processor(ann["question_text"])
        history_text = self.text_processor(ann["history_text"])
        history_text = history_text.replace('问题：', '##MASK##').replace('回复：', '##MASK##').split('##MASK##')
        history_text = history_text[1:] if history_text[0] == '' else history_text
        history_text = [(history_text[i], history_text[i+1]) for i in range(0, len(history_text), 2)]

        input_text = ''
        for i, (old_query, response) in enumerate(history_text):
                input_text += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        input_text += "[Round {}]\n问：{}\n答：".format(len(history_text), source_text)
    

        image_path = ann["image_path"].split('/')[-1]
        if 'train' in image_path:
            image_path = os.path.join(self.vis_root, 'train2014', image_path)
        else:
            image_path = os.path.join(self.vis_root, 'val2014', image_path)

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
            "text_input": input_text,
            "speech_input": speech_feats,
            "text_output": target_text,
            "task_id": 4,
        }
    
