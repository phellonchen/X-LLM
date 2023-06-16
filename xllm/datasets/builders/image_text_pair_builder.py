"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from xllm.common.registry import registry

from xllm.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from xllm.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset
from xllm.datasets.datasets.laion_dataset import LaionDataset
from xllm.datasets.datasets.image_text_pair_datasets_generation import ImageTextPairDatasetGeneration
from xllm.datasets.datasets.image_text_pair_datasets_multi import ImageTextPairDatasetMulti
from xllm.datasets.datasets.image_text_pair_datasets_generation_multi import ImageTextPairDatasetGenerationMulti
from xllm.datasets.datasets.image_text_pair_datasets_generation_caption import ImageTextPairDatasetGenerationCaption
from xllm.datasets.datasets.image_text_pair_datasets_generation_multi_caption import ImageTextPairDatasetGenerationMultiCaption
from xllm.datasets.datasets.image_text_pair_datasets_generation_vqa_vd import ImageTextPairDatasetGenerationVQAVD, ImageTextPairDatasetGenerationVQAVDEval
from xllm.datasets.datasets.image_text_pair_datasets_generation_text import ImageTextPairDatasetGenerationText
from xllm.datasets.datasets.image_text_pair_datasets_generation_dialog import ImageTextPairDatasetGenerationDialog

@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m.yaml"
    }


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}

@registry.register_builder("coco_caption_zh")
class COCOCaptionZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_cap_zh.yaml"}

@registry.register_builder("ai_challenge_zh")
class AiChallengeZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ai_challenge/defaults_cap_zh.yaml"}

@registry.register_builder("flickr30k_zh")
class Flickr30kZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults_zh.yaml"}

@registry.register_builder("muge_zh")
class MugeZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/muge/defaults_cap_zh.yaml"}

@registry.register_builder("vg_caption_zh")
class VGCaptionZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_zh.yaml"}

@registry.register_builder("sbu_caption_zh")
class SBUaptionZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults_zh.yaml"}

@registry.register_builder("chinese_food_zh")
class ChinesefoodZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/chinese_food/defaults_zh.yaml"}

@registry.register_builder("cc3m_caption_zh")
class CC3MCaptionZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc3m/defaults_cap_zh.yaml"}

@registry.register_builder("wukong_caption_zh")
class WukongZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/wukong/defaults_cap_zh.yaml"}

@registry.register_builder("coco_caption_zh_g")
class COCOCaptionZhGCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_cap_zh_g.yaml"}

@registry.register_builder("ai_challenge_zh_g")
class AiChallengeZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ai_challenge/defaults_cap_zh_g.yaml"}

@registry.register_builder("flickr30k_zh_g")
class Flickr30kZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults_zh_g.yaml"}

@registry.register_builder("muge_zh_g")
class MugeZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMulti

    DATASET_CONFIG_DICT = {"default": "configs/datasets/muge/defaults_cap_zh_g.yaml"}

@registry.register_builder("cc3m_caption_zh_g")
class CC3MCaptionZhGBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGeneration

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc3m/defaults_cap_zh_g.yaml"}

@registry.register_builder("wukong_caption_zh_g")
class WukongZhBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGeneration

    DATASET_CONFIG_DICT = {"default": "configs/datasets/wukong/defaults_cap_zh_g.yaml"}

@registry.register_builder("vg_caption_zh_g")
class VGCaptionZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGeneration

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_zh_g.yaml"}

@registry.register_builder("sbu_caption_zh_g")
class SBUCaptionZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGeneration

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults_zh_g.yaml"}

@registry.register_builder("chinese_food_zh_g")
class ChineseFoddZhCaptionBuilder_g(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGeneration

    DATASET_CONFIG_DICT = {"default": "configs/datasets/chinese_food/defaults_zh_g.yaml"}


@registry.register_builder("coco_caption_zh_cap")
class COCOCaptionZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMultiCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_cap_zh_cap.yaml"}

@registry.register_builder("ai_challenge_zh_cap")
class AiChallengeZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMultiCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ai_challenge/defaults_cap_zh_cap.yaml"}

@registry.register_builder("flickr30k_zh_cap")
class Flickr30kZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMultiCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults_zh_cap.yaml"}

@registry.register_builder("muge_zh_cap")
class MugeZhCaptionZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationMultiCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/muge/defaults_cap_zh_cap.yaml"}

@registry.register_builder("cc3m_caption_zh_cap")
class CC3MCaptionZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc3m/defaults_cap_zh_cap.yaml"}

@registry.register_builder("wukong_caption_zh_cap")
class WukongCapZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/wukong/defaults_cap_zh_cap.yaml"}

@registry.register_builder("vg_caption_zh_cap")
class VGCaptionZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_zh_cap.yaml"}

@registry.register_builder("sbu_caption_zh_cap")
class SBUCaptionZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults_zh_cap.yaml"}

@registry.register_builder("chinese_food_zh_cap")
class ChineseFoodCapZhCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationCaption

    DATASET_CONFIG_DICT = {"default": "configs/datasets/chinese_food/defaults_zh_cap.yaml"}


@registry.register_builder("vd")
class VdZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationVQAVD
    eval_dataset_cls = ImageTextPairDatasetGenerationVQAVDEval
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vqa_vd/defaults_zh_cap.yaml"}

@registry.register_builder("vqa_zh")
class VqaZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationVQAVD
    eval_dataset_cls = ImageTextPairDatasetGenerationVQAVDEval
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vqa_vd/defaults_zh_cap_vqa.yaml"}

@registry.register_builder("gqa_zh")
class GqaZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationVQAVD
    eval_dataset_cls = ImageTextPairDatasetGenerationVQAVDEval
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vqa_vd/defaults_zh_cap_gqa.yaml"}

@registry.register_builder("baike_qa")
class BaikeZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationText
    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_qa/defaults_zh_cap_baike_qa.yaml"}


@registry.register_builder("wiki")
class WikiZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationText
    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_qa/defaults_zh_cap_wiki.yaml"}

@registry.register_builder("pclue")
class PclueZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationText
    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_qa/defaults_zh_cap_pclue.yaml"}

@registry.register_builder("dialog")
class DialogZhCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDatasetGenerationDialog
    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_qa/defaults_zh_cap_dialog.yaml"}

@registry.register_builder("laion2B_multi")
class Laion2BMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "xllm/configs/datasets/laion/defaults_2B_multi.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
