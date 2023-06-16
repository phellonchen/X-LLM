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


class ImageTextPairDatasetMulti(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # coco, Flickr30k_zh, MUGE_zh, ai_challenge
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        if ann['dataset'] in ['Flickr30k_zh', 'MUGE_zh']:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(ann["image"]))).convert("RGB")
        else:
            image_path = os.path.join(self.vis_root, ann["image"])
            # print(image_path)
            image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
   
        caption = random.choice(ann["caption"])
        caption = self.text_processor(caption)
    

        return {"image": image, "text_input": caption}
