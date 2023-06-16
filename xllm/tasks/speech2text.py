"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import json
import os
import logging
import numpy as np
import editdistance

from xllm.common.dist_utils import main_process
from xllm.common.registry import registry
from xllm.tasks.base_task import BaseTask


@registry.register_task("speech2text")
class Speech2TextTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []
        hyps = model.asr_generate(
            samples,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        text_outputs = samples["text_output"]
        for idx,  hyp in enumerate(hyps):
            results.append({"hyp": hyp, 'ref': text_outputs[idx]})

        total_err_num = 0
        total_ref_len = 0
        text_outputs = samples["text_output"]
        for hyp, ref in zip(hyps, text_outputs):
            logging.info(f"REF: {ref}")
            logging.info(f"HYP: {hyp}")
            errors = editdistance.eval(ref, hyp)
            error_rate = errors / len(list(ref))
            logging.info(f"Error Rate: {error_rate}, Error Num: {errors}, Reference Length: {len(list(ref))}")
            logging.info("_________________________________________________________________")
            total_err_num += errors
            total_ref_len += len(list(ref))
        batch_error_rate = total_err_num / total_ref_len
        logging.info(f"Batch statistics: Error Rate = {batch_error_rate}")

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        metrics = self._report_metrics(
                eval_result_file=val_result, split_name=split_name
            )

        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=val_result, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics
    
    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        total_err_num = 0
        total_ref_len = 0
 
        for item in eval_result_file:
            hyp = item['hyp']
            if hyp == 'Error':
                continue
            ref = item['ref']
            logging.info(f"REF: {ref}")
            logging.info(f"HYP: {hyp}")
            errors = editdistance.eval(ref, hyp)
            error_rate = errors / len(list(ref))
            logging.info(f"Error Rate: {error_rate}, Error Num: {errors}, Reference Length: {len(list(ref))}")
            logging.info("_________________________________________________________________")
            total_err_num += errors
            total_ref_len += len(list(ref))
        batch_error_rate = total_err_num / total_ref_len
        logging.info(f"Batch statistics: Error Rate = {batch_error_rate}")

        coco_res = {'CER: ': batch_error_rate}
        coco_res["agg_metrics"] = batch_error_rate

        return coco_res
