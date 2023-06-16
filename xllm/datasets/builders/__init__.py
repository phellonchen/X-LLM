"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from xllm.datasets.builders.base_dataset_builder import load_dataset_config
from xllm.datasets.builders.caption_builder import (
    COCOCapBuilder,
    MSRVTTCapBuilder,
    MSVDCapBuilder,
    VATEXCapBuilder,
)
from xllm.datasets.builders.image_text_pair_builder import (
    ConceptualCaption12MBuilder,
    ConceptualCaption3MBuilder,
    VGCaptionBuilder,
    SBUCaptionBuilder,
    # ZH
    COCOCaptionZhCaptionBuilder,
    AiChallengeZhCaptionBuilder,
    Flickr30kZhCaptionBuilder,
    MugeZhCaptionBuilder,
    VGCaptionZhCaptionBuilder,
    SBUaptionZhCaptionBuilder,
    ChinesefoodZhCaptionBuilder,
    CC3MCaptionZhBuilder,
    WukongZhBuilder,
    # g
    COCOCaptionZhGCaptionBuilder_g,
    AiChallengeZhCaptionBuilder_g,
    Flickr30kZhCaptionBuilder_g,
    MugeZhCaptionBuilder_g,
    CC3MCaptionZhGBuilder_g,
    WukongZhBuilder_g,
    VGCaptionZhCaptionBuilder_g,
    SBUCaptionZhCaptionBuilder_g,
    ChineseFoddZhCaptionBuilder_g,
    # cap
    COCOCaptionZhCapBuilder,
    AiChallengeZhCapBuilder,
    Flickr30kZhCapBuilder,
    MugeZhCaptionZhCapBuilder,
    CC3MCaptionZhCapBuilder,
    WukongCapZhBuilder,
    VGCaptionZhCapBuilder,
    SBUCaptionZhCapBuilder,
    ChineseFoodCapZhCapBuilder,
    VdZhCaptionBuilder,
    VqaZhCaptionBuilder,
    GqaZhCaptionBuilder,
    BaikeZhCaptionBuilder,
    WikiZhCaptionBuilder,
    PclueZhCaptionBuilder,
    DialogZhCaptionBuilder
    
)
from xllm.datasets.builders.classification_builder import (
    NLVRBuilder,
    SNLIVisualEntailmentBuilder,
)
from xllm.datasets.builders.imagefolder_builder import ImageNetBuilder
from xllm.datasets.builders.video_qa_builder import MSRVTTQABuilder, MSVDQABuilder
from xllm.datasets.builders.vqa_builder import (
    COCOVQABuilder,
    OKVQABuilder,
    VGVQABuilder,
    GQABuilder,
)
from xllm.datasets.builders.retrieval_builder import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
    Flickr30kBuilder,
)
from xllm.datasets.builders.dialogue_builder import AVSDDialBuilder
from xllm.datasets.builders.asr_builder import Aishell2AsrZhBuilder, VsdialMAsrZhBuilder
# from xllm.datasets.builders.vsd_builder import VsdialZhBuilder
from xllm.datasets.builders.instruction_builder import VLInstructionBuilder, ASRInstructionBuilder, MASRInstructionBuilder, VSDInstructionBuilder, VideoInstructionBuilder, VideoCaptionBuilder

from xllm.common.registry import registry

__all__ = [
    "COCOCapBuilder",
    "COCORetrievalBuilder",
    "COCOVQABuilder",
    "ConceptualCaption12MBuilder",
    "ConceptualCaption3MBuilder",
    "DiDeMoRetrievalBuilder",
    "Flickr30kBuilder",
    "GQABuilder",
    "ImageNetBuilder",
    "MSRVTTCapBuilder",
    "MSRVTTQABuilder",
    "MSRVTTRetrievalBuilder",
    "MSVDCapBuilder",
    "MSVDQABuilder",
    "NLVRBuilder",
    "OKVQABuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "VATEXCapBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "AVSDDialBuilder",
    # ZH
    "COCOCaptionZhCaptionBuilder",
    "AiChallengeZhCaptionBuilder",
    "Flickr30kZhCaptionBuilder",
    "MugeZhCaptionBuilder",
    "VGCaptionZhCaptionBuilder",
    "SBUaptionZhCaptionBuilder",
    "ChinesefoodZhCaptionBuilder",
    "CC3MCaptionZhBuilder",
    "WukongZhBuilder",
    # g
    "COCOCaptionZhGCaptionBuilder_g",
    "AiChallengeZhCaptionBuilder_g",
    "Flickr30kZhCaptionBuilder_g",
    "MugeZhCaptionBuilder_g",
    "CC3MCaptionZhGBuilder_g",
    "WukongZhBuilder_g",
    "VGCaptionZhCaptionBuilder_g",
    "SBUCaptionZhCaptionBuilder_g",
    "ChineseFoddZhCaptionBuilder_g",
    # cap
    "COCOCaptionZhCapBuilder",
    "AiChallengeZhCapBuilder",
    "Flickr30kZhCapBuilder",
    "MugeZhCaptionZhCapBuilder",
    "CC3MCaptionZhCapBuilder",
    "WukongCapZhBuilder",
    "VGCaptionZhCapBuilder",
    "SBUCaptionZhCapBuilder",
    "ChineseFoodCapZhCapBuilder",
    "VdZhCaptionBuilder",
    "VqaZhCaptionBuilder",
    "GqaZhCaptionBuilder",
    "BaikeZhCaptionBuilder",
    "WikiZhCaptionBuilder",
    "PclueZhCaptionBuilder",
    "DialogZhCaptionBuilder",
    "Aishell2AsrZhBuilder",
    # instruction
    "VLInstructionBuilder", 
    "ASRInstructionBuilder", 
    "MASRInstructionBuilder", 
    "VSDInstructionBuilder",
    "VideoInstructionBuilder",
    "VsdialMAsrZhBuilder",
    "VideoCaptionBuilder",
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
