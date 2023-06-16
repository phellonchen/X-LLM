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


class ImageTextPairDatasetGeneration(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        if 'food' in ann['dataset']:
            ann['image'] = random.choice(ann['image'])
            image_path = os.path.join(self.vis_root, ann["image"])
        else:
            image_path = ann["image"]


        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        caption = self.text_processor(ann["caption"]).strip('。')

        source_text, target_text = self.corrupt_prefix_zh(caption)
        source_text = "用户：" + source_text + "\\n小元："

        return {"image": image, "text_input": '[gMASK]', "text_output": caption}
        # return {"image": image, "text_input": source_text, "text_output": target_text}
    
    def corrupt_prefix(self, input_text):
        """T5-style Prefix Language Modelings
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

        tokens = input_text.split()

        n_tokens = len(tokens)
        split = random.randint(1, n_tokens-1)
        source_text = " ".join(tokens[:split])
        target_text = " ".join(tokens[split:])

        return source_text, target_text
    
    def corrupt_prefix_zh(self, input_text):
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

        n_tokens = len(tokens)
        if n_tokens > 1:
            split = random.randint(1, n_tokens-1)
            split = min(split, 30)
            source_text = "这张图描述了什么？" + "".join(tokens[:split])
            target_text = "".join(tokens[split:][:30])
        else:
            source_text, target_text = "这张图描述了什么？", "".join(tokens)

        return source_text, target_text
