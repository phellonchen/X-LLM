"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from xllm.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import numpy as np

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


class VLInstructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompt = [
            '请详细描述这张图片。',
            '请查看这张图片并描述您注意到的内容。',
            '请提供这张图片的详细描述',
            '你能为我描述这张图片的内容吗？',
        ]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index] 
        image_path = os.path.join(self.vis_root, ann["image_id"]+".jpg")
      
        image = Image.open(image_path).convert("RGB")

        speech_feats = np.load("/raid/cfl/data/all_audio_cif_feats/split_shards/train-vsdial/250362.npy") 

        image = self.vis_processor(image)

        caption = self.text_processor(ann["caption_zh"])
        text_input = '问：' + random.choice(self.prompt) + '\n答：'

        max_length = 20
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
            )   # 

        return {"image": image, "text_input": text_input, "text_output": caption, "task_id": 1, "speech_input": speech_feats}
      

