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
import base64
from io import BytesIO

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


class ImageTextPairDatasetGenerationMultiCaption(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.prompt = ['你能看见什么？','请描述一下这张图片','这幅图片的内容是什么？','这张图片上有什么','从这张图片上你能获得什么信息？', '图里有什么', '这是什么图片','图片描绘了什么？', '用语言描述一下这幅图片']
        # self.prompt = ['请描述一下这张图片']

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        if ann['dataset'] in ['Flickr30k_zh', 'MUGE_zh']:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(ann["image"]))).convert("RGB")
        else:
            image_path = os.path.join(self.vis_root, ann["image"])
            image = Image.open(image_path).convert("RGB")
            
        image = self.vis_processor(image)
   
        # caption = random.choice(ann["caption"])
        caption = ann["caption"][0]

        caption = self.text_processor(caption)
        target_text = caption
 
        source_text = "用户：" + "请描述一下这张图片" + " \\n小元："

        return {"image": image, "text_input": source_text, "text_output": target_text}
    
    def corrupt_prefix(self, input_text):
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
            source_text = "".join(tokens[:split])
            target_text = "".join(tokens[split:])
        else:
            source_text, target_text = "这张图描述了什么？", "".join(tokens)

        return source_text, target_text
