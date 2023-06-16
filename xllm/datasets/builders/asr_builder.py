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
from xllm.datasets.datasets.masr_datasets import MSpeech2TextDataset

@registry.register_builder("aishell2_asr_zh")
class Aishell2AsrZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = Speech2TextDataset
    eval_dataset_cls = Speech2TextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aishell2/defaults_asr_zh.yaml"
    }

@registry.register_builder("js_30k_asr_zh")
class JS30kAsrZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = Speech2TextDataset
    eval_dataset_cls = Speech2TextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/js_paired_data/defaults_js30k_asr_zh.yaml"
    }

@registry.register_builder("vsdial_asr_zh")
class VsdialAsrZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = Speech2TextDataset
    eval_dataset_cls = Speech2TextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsdial/defaults_vsdial_asr_zh.yaml"
    }

@registry.register_builder("vsdial_masr_zh")
class VsdialMAsrZhBuilder(BaseDatasetBuilder):
    train_dataset_cls = MSpeech2TextDataset
    eval_dataset_cls = MSpeech2TextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsdial/defaults_vsdial_masr_zh.yaml"
    }