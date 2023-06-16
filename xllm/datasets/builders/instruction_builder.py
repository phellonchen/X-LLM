"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from xllm.common.registry import registry
from xllm.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from xllm.datasets.datasets.asr_datasets import Speech2TextDataset
from xllm.datasets.datasets.vl_instruction import VLInstructionDataset
from xllm.datasets.datasets.asr_instruction import ASRInstructionDataset
from xllm.datasets.datasets.masr_instruction import MASRInstructionDataset
from xllm.datasets.datasets.vsd_instruction import VSDIntructionDataset
from xllm.datasets.datasets.video_instruction import VideoInstructionDataset
from xllm.datasets.datasets.video_caption import VideoCaptionDataset

@registry.register_builder("vl_instruction_zh")
class VLInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = VLInstructionDataset
    eval_dataset_cls = VLInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruction/defaults_vl_zh.yaml"
    }

@registry.register_builder("asr_instruction_zh")
class ASRInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ASRInstructionDataset
    eval_dataset_cls = ASRInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruction/defaults_asr_zh.yaml"
    }

@registry.register_builder("masr_instruction_zh")
class MASRInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = MASRInstructionDataset
    eval_dataset_cls = MASRInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruction/defaults_masr_zh.yaml"
    }

@registry.register_builder("vsd_instruction_zh")
class VSDInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSDIntructionDataset
    eval_dataset_cls = VSDIntructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruction/defaults_vsd_zh.yaml"
    }

@registry.register_builder("video_instruction_zh")
class VideoInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoInstructionDataset
    eval_dataset_cls = VideoInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/activitycaps/defaults_video_zh.yaml"
    }

@registry.register_builder("video_zh")
class VideoCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/activitycaps/defaults_video_zh_activitycaps.yaml"
    }